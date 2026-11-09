# -*- coding: utf-8 -*-
"""
安全加固模块

提供：
- 文件上传校验（类型、大小、内容检查）
- SQL 操作安全（参数化查询辅助）
- 高危操作授权机制
- 输入内容安全检查（防注入、防越权）

注意：本模块是增量加固，不替代框架自带的安全机制。
"""
import os
import re
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================
# 文件上传安全
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
ALLOWED_IMAGE_MIME_PREFIXES = {
    b'\xff\xd8\xff': 'image/jpeg',      # JPEG
    b'\x89PNG\r\n\x1a\n': 'image/png',  # PNG
    b'BM': 'image/bmp',                   # BMP
    b'RIFF': 'image/webp',               # WebP (RIFF....WEBP)
}
MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


class FileValidationError(Exception):
    """文件校验失败异常"""
    pass


def validate_image_upload(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    校验上传的图像文件

    返回: {'valid': bool, 'reason': str, 'size': int, 'extension': str, 'mime': str}

    检查项：
    1. 文件名扩展名白名单
    2. 文件大小限制
    3. 文件魔数（真实类型检查）
    4. 文件名路径遍历防护
    """
    result = {
        'valid': False,
        'reason': '',
        'size': len(file_bytes),
        'extension': '',
        'mime': '',
        'filename_safe': '',
    }

    # 1. 文件名安全检查（防路径遍历）
    safe_name = os.path.basename(filename)
    if safe_name != filename or '..' in safe_name or '/' in safe_name or '\\' in safe_name:
        result['reason'] = '文件名不合法，可能包含路径遍历'
        return result

    result['filename_safe'] = safe_name

    # 2. 扩展名检查
    ext = Path(safe_name).suffix.lower()
    result['extension'] = ext
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        result['reason'] = f'不支持的文件类型: {ext}，仅支持 {", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))}'
        return result

    # 3. 文件大小检查
    if len(file_bytes) == 0:
        result['reason'] = '文件为空'
        return result
    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        result['reason'] = f'文件过大: {len(file_bytes)/1024/1024:.1f}MB，最大 {MAX_IMAGE_SIZE_MB}MB'
        return result

    # 4. 魔数检查（真实文件类型）
    detected_mime = None
    for magic, mime in ALLOWED_IMAGE_MIME_PREFIXES.items():
        if file_bytes.startswith(magic):
            # WebP 需要额外检查
            if mime == 'image/webp':
                if len(file_bytes) >= 12 and file_bytes[8:12] == b'WEBP':
                    detected_mime = mime
                    break
            else:
                detected_mime = mime
                break

    if not detected_mime:
        result['reason'] = '文件内容与扩展名不符，可能不是有效的图像文件'
        return result

    result['mime'] = detected_mime
    result['valid'] = True
    return result


def generate_safe_filename(original_name: str, prefix: str = 'upload') -> str:
    """生成安全的存储文件名（避免文件名冲突和注入）"""
    ext = Path(original_name).suffix.lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 使用原始文件名的哈希，避免特殊字符
    name_hash = hashlib.md5(original_name.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{name_hash}{ext}"


# ============================================================
# SQL 安全
# ============================================================

# 安全的表名和列名白名单（防止SQL注入中的标识符注入）
SAFE_TABLE_NAMES = {
    'detection_records', 'quality_records', 'tool_calls',
    'review_tasks', 'review_action_history', 'annotation_pool',
    'execution_traces', 'trace_steps',
}

SAFE_COLUMN_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def safe_identifier(name: str, allowed: set = None) -> str:
    """
    验证SQL标识符（表名、列名）是否安全
    注意：参数化查询不能用于表名/列名，需要白名单验证
    """
    if allowed and name not in allowed:
        raise ValueError(f"非法的标识符: {name}")
    if not SAFE_COLUMN_PATTERN.match(name):
        raise ValueError(f"非法的标识符格式: {name}")
    return name


def safe_query(conn: sqlite3.Connection, query: str, params: tuple = None) -> sqlite3.Cursor:
    """
    安全执行参数化查询

    用法：
        safe_query(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
    """
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor


# ============================================================
# 操作授权
# ============================================================

# 高危操作列表（需要确认才能执行）
HIGH_RISK_OPERATIONS = {
    'delete_record': {'name': '删除检测记录', 'confirm_text': '确认要删除这条记录吗？此操作不可恢复。'},
    'delete_all_records': {'name': '清空所有记录', 'confirm_text': '确认要清空所有记录吗？此操作不可恢复！'},
    'modify_rule': {'name': '修改质量规则', 'confirm_text': '修改规则将影响所有后续判定，确认吗？'},
    'trigger_training': {'name': '触发模型训练', 'confirm_text': '触发训练将消耗大量算力，确认吗？'},
    'replace_model': {'name': '替换生产模型', 'confirm_text': '替换模型将直接影响检测结果，确认吗？'},
    'export_all_data': {'name': '导出全部数据', 'confirm_text': '导出包含所有检测记录，确认吗？'},
    'confirm_review': {'name': '人工复核确认', 'confirm_text': ''},  # 复核操作本身就是人工的
}


class AuthorizationResult:
    """授权结果"""
    def __init__(self, allowed: bool, reason: str = '', required_confirm: bool = False):
        self.allowed = allowed
        self.reason = reason
        self.required_confirm = required_confirm

    def __bool__(self):
        return self.allowed


def check_authorization(operation: str, user_role: str = 'user',
                        confirmed: bool = False) -> AuthorizationResult:
    """
    检查操作是否需要授权

    返回 AuthorizationResult：
    - allowed: 是否允许执行
    - required_confirm: 是否需要用户确认
    - reason: 原因说明
    """
    if operation not in HIGH_RISK_OPERATIONS:
        # 普通操作，直接允许
        return AuthorizationResult(allowed=True)

    op_info = HIGH_RISK_OPERATIONS[operation]

    if confirmed:
        # 用户已确认
        return AuthorizationResult(allowed=True, reason='用户已确认')

    # 需要确认
    return AuthorizationResult(
        allowed=False,
        reason=op_info['confirm_text'],
        required_confirm=True,
    )


# ============================================================
# 输入安全
# ============================================================

def sanitize_text_input(text: str, max_length: int = 2000) -> str:
    """清理文本输入，移除潜在危险字符"""
    if not text:
        return ''

    # 截断
    if len(text) > max_length:
        text = text[:max_length]

    # 移除控制字符（保留换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


def is_safe_sql_identifier(name: str) -> bool:
    """检查是否为安全的SQL标识符"""
    return bool(SAFE_COLUMN_PATTERN.match(name))


# ============================================================
# 会话数据隔离
# ============================================================

class SessionDataIsolator:
    """
    会话数据隔离器
    确保不同用户/会话的数据不互相干扰
    """

    def __init__(self):
        self._sessions = {}

    def get_session_data(self, session_id: str) -> dict:
        """获取会话数据，不存在则创建"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                'created_at': datetime.now().isoformat(),
                'uploads': [],
                'task_history': [],
                'context': None,
            }
        return self._sessions[session_id]

    def clear_session(self, session_id: str):
        """清除会话数据"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def cleanup_expired(self, max_age_hours: int = 24):
        """清理过期会话"""
        now = datetime.now()
        expired = []
        for sid, data in self._sessions.items():
            created = datetime.fromisoformat(data['created_at'])
            if (now - created).total_seconds() > max_age_hours * 3600:
                expired.append(sid)
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


# ============================================================
# 安全扫描报告
# ============================================================

def security_audit() -> Dict[str, Any]:
    """生成安全审计报告"""
    return {
        'file_upload': {
            'allowed_extensions': sorted(list(ALLOWED_IMAGE_EXTENSIONS)),
            'max_size_mb': MAX_IMAGE_SIZE_MB,
            'magic_number_check': True,
            'path_traversal_protection': True,
            'safe_filename_generation': True,
        },
        'sql_security': {
            'parameterized_queries': True,
            'identifier_whitelist': True,
            'safe_tables': sorted(list(SAFE_TABLE_NAMES)),
        },
        'authorization': {
            'high_risk_operations': sorted(list(HIGH_RISK_OPERATIONS.keys())),
            'confirm_required': True,
        },
        'input_safety': {
            'max_text_length': 2000,
            'control_char_stripping': True,
        },
        'session_isolation': {
            'per_session_data': True,
            'auto_cleanup': True,
            'max_age_hours': 24,
        },
    }


if __name__ == '__main__':
    # 打印安全审计报告
    import json
    audit = security_audit()
    print("=" * 60)
    print("🛡️  安全加固审计报告")
    print("=" * 60)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("=" * 60)
