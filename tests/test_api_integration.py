# -*- coding: utf-8 -*-
"""FastAPI 接口集成测试"""
import sys
sys.path.insert(0, '.')
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"
TEST_IMAGE = r"data\raw\NEU-DET\IMAGES\crazing_1.jpg"
TEST_IMAGES = [
    r"data\raw\NEU-DET\IMAGES\crazing_1.jpg",
    r"data\raw\NEU-DET\IMAGES\inclusion_1.jpg",
    r"data\raw\NEU-DET\IMAGES\patches_1.jpg",
]

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    try:
        func()
        print(f"✅ {name} 通过")
        passed += 1
    except Exception as e:
        print(f"❌ {name} 失败: {e}")
        failed += 1
        import traceback
        traceback.print_exc()

def check_response(resp):
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"
    data = resp.json()
    assert data['code'] == 0, f"API error: {data.get('message')}"
    return data['data']

# 1. 健康检查
def test_health():
    resp = requests.get(f"{BASE_URL}/api/health")
    d = check_response(resp)
    print(f"  status: {d['status']}")
    print(f"  model_loaded: {d['model_loaded']}")
    print(f"  llm_available: {d['llm_available']}")
    assert d['status'] == 'ok'
    assert d['model_loaded'] == True

# 2. 系统配置
def test_config():
    resp = requests.get(f"{BASE_URL}/api/config")
    d = check_response(resp)
    print(f"  model_version: {d['model_version']}")
    print(f"  rule_id: {d['rule_id']}")
    print(f"  llm_provider: {d['llm_provider']}")
    print(f"  classes: {len(d['class_names'])} 类")
    assert d['model_version']
    assert len(d['class_names']) == 6

# 3. 单图检测
def test_detect():
    with open(TEST_IMAGE, 'rb') as f:
        resp = requests.post(
            f"{BASE_URL}/api/detect",
            files={'image': ('crazing_1.jpg', f, 'image/jpeg')},
        )
    d = check_response(resp)
    print(f"  image_name: {d['image_name']}")
    print(f"  detections: {d['detection_count']} 个")
    q = d['quality_result']
    print(f"  quality: {q['quality']}")
    print(f"  score: {q['score']}")
    print(f"  review_id: {d.get('review_id', 'None')}")
    assert d['detection_count'] > 0
    assert q['quality'] in ('PASS', 'REVIEW', 'FAIL')
    # FAIL 应该自动创建了复核任务
    if q['quality'] == 'FAIL':
        assert d.get('review_id'), "FAIL 应自动创建复核任务"

# 4. 质量判定（给定检测结果）
def test_quality_judge():
    # 先用检测接口拿到检测结果
    with open(TEST_IMAGE, 'rb') as f:
        resp = requests.post(
            f"{BASE_URL}/api/detect",
            files={'image': ('crazing_1.jpg', f, 'image/jpeg')},
        )
    det_data = check_response(resp)

    # 再调用判定接口
    judge_req = {
        'detections': det_data['detections'],
        'image_width': det_data['image_info']['width'],
        'image_height': det_data['image_info']['height'],
    }
    resp2 = requests.post(f"{BASE_URL}/api/quality/judge", json=judge_req)
    d = check_response(resp2)
    print(f"  quality: {d['quality']}")
    print(f"  score: {d['score']}")
    assert d['quality'] in ('PASS', 'REVIEW', 'FAIL')

# 5. 批次检测
def test_batch_detect():
    files = []
    for i, path in enumerate(TEST_IMAGES):
        name = os.path.basename(path)
        files.append(('images', (name, open(path, 'rb'), 'image/jpeg')))
    try:
        resp = requests.post(f"{BASE_URL}/api/batch/detect", files=files)
    finally:
        for _, (_, f, _) in files:
            f.close()
    d = check_response(resp)
    print(f"  batch_id: {d['batch_id']}")
    print(f"  total_planned: {d['total_planned']}")
    print(f"  total_success: {d['total_success']}")
    print(f"  batch_quality: {d['batch_quality']}")
    print(f"  images: {len(d['image_results'])} 张")
    for img in d['image_results']:
        print(f"    - {img['image_name']}: {img['quality']} ({img['detection_count']}个)")
    assert d['total_planned'] == 3
    assert d['total_success'] == 3

# 6. 复核任务列表
def test_review_list():
    resp = requests.get(f"{BASE_URL}/api/review/tasks")
    d = check_response(resp)
    print(f"  total: {d['total']}")
    print(f"  tasks: {len(d['tasks'])} 个")
    if d['tasks']:
        t = d['tasks'][0]
        print(f"  第一个: {t['review_id']} - {t['image_name']} ({t['original_quality']})")
    assert 'tasks' in d

# 7. 复核任务详情 + 确认
def test_review_detail_and_confirm():
    # 先获取列表
    resp = requests.get(f"{BASE_URL}/api/review/tasks?status=pending")
    d = check_response(resp)
    if not d['tasks']:
        print("  ⚠️  没有待处理任务，跳过确认测试")
        return
    task = d['tasks'][0]
    rid = task['review_id']
    print(f"  选中任务: {rid}")

    # 详情
    resp2 = requests.get(f"{BASE_URL}/api/review/tasks/{rid}")
    detail = check_response(resp2)
    t = detail['task']
    actions = detail.get('action_history', [])
    print(f"  详情: {t['image_name']} - {t['original_quality']}")
    print(f"  操作历史: {len(actions)} 条")

    # 确认
    resp3 = requests.post(
        f"{BASE_URL}/api/review/tasks/{rid}/confirm",
        json={'actor': 'test_user', 'role': 'inspector', 'note': 'API测试确认'},
    )
    confirm_data = check_response(resp3)
    print(f"  确认结果: {confirm_data['success'] if 'success' in confirm_data else confirm_data.get('status')}")

    # 验证状态变更
    resp4 = requests.get(f"{BASE_URL}/api/review/tasks/{rid}")
    detail2 = check_response(resp4)
    new_status = detail2['task']['status']
    print(f"  新状态: {new_status}")
    assert new_status == 'confirmed'

# 8. Agent 对话（form-data + 文件上传）
def test_agent_chat():
    with open(TEST_IMAGE, 'rb') as f:
        resp = requests.post(
            f"{BASE_URL}/api/agent/chat",
            data={'message': '检测这张图的缺陷并判定质量'},
            files={'image': ('crazing_1.jpg', f, 'image/jpeg')},
        )
    d = check_response(resp)
    print(f"  task_id: {d['task_id']}")
    print(f"  final_quality: {d['final_quality']}")
    print(f"  steps: {d['total_steps']}")
    reply = d.get('reply', '') or d.get('response', '')
    print(f"  reply: {reply[:100]}...")
    assert d['final_quality'] in ('PASS', 'REVIEW', 'FAIL')
    assert d['total_steps'] >= 2

# 9. 生成批次报告
def test_batch_report():
    # 先做批次检测
    files = []
    for i, path in enumerate(TEST_IMAGES):
        name = os.path.basename(path)
        files.append(('images', (name, open(path, 'rb'), 'image/jpeg')))
    try:
        resp = requests.post(f"{BASE_URL}/api/batch/detect", files=files)
    finally:
        for _, (_, f, _) in files:
            f.close()
    batch_data = check_response(resp)
    bid = batch_data['batch_id']
    print(f"  batch_id: {bid}")

    # 生成报告
    resp2 = requests.post(f"{BASE_URL}/api/batch/{bid}/report")
    d = check_response(resp2)
    print(f"  report_id: {d['report_id']}")
    print(f"  report_type: {d['report_type']}")
    assert d['report_id']

# 10. 下载报告
def test_download_report():
    # 先生成一个批次报告
    files = []
    for i, path in enumerate(TEST_IMAGES):
        name = os.path.basename(path)
        files.append(('images', (name, open(path, 'rb'), 'image/jpeg')))
    try:
        resp = requests.post(f"{BASE_URL}/api/batch/detect", files=files)
    finally:
        for _, (_, f, _) in files:
            f.close()
    batch_data = check_response(resp)
    bid = batch_data['batch_id']

    resp2 = requests.post(f"{BASE_URL}/api/batch/{bid}/report")
    report_data = check_response(resp2)
    rid = report_data['report_id']
    print(f"  report_id: {rid}")

    # 下载
    resp3 = requests.get(f"{BASE_URL}/api/report/{rid}/download")
    assert resp3.status_code == 200
    ct = resp3.headers.get('content-type', '')
    assert 'vnd.openxmlformats' in ct or 'docx' in ct or 'msword' in ct
    size = len(resp3.content)
    print(f"  下载大小: {size} bytes")
    assert size > 1000, "报告文件太小"

# 运行所有测试
test("健康检查", test_health)
test("系统配置", test_config)
test("单图检测", test_detect)
test("质量判定", test_quality_judge)
test("批次检测", test_batch_detect)
test("复核任务列表", test_review_list)
test("复核详情+确认", test_review_detail_and_confirm)
test("Agent对话", test_agent_chat)
test("批次报告生成", test_batch_report)
test("报告下载", test_download_report)

print(f"\n{'='*60}")
print(f"API 测试: {passed} 通过, {failed} 失败, 共 {passed+failed} 项")
print(f"{'='*60}")
if failed == 0:
    print("🎉 所有 API 测试通过！")
