# -*- coding: utf-8 -*-
"""
复核管理页面 —— 人机复核工作流

功能：
1. 复核任务列表（支持状态筛选：待处理/已确认/已驳回/已修正/已升级）
2. 任务详情查看（原始检测结果、原始判定）
3. 复核操作：确认（inspector角色）/ 驳回并修正（inspector角色）
4. RBAC 角色选择（操作员/质检员/主管），权限验证
5. 操作日志查看
"""

import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime

from models.agent.review_workflow import (
    ReviewWorkflow,
    ReviewStatus,
    ROLE_PERMISSIONS,
    AuthorizationError,
)


# 角色中文名称映射
ROLE_LABELS = {
    'operator': '👤 操作员',
    'inspector': '🔍 质检员',
    'supervisor': '👔 主管',
    'viewer': '👁️ 查看者',
}

# 状态中文名称映射
STATUS_LABELS = {
    'pending': '⏳ 待处理',
    'in_progress': '🔄 处理中',
    'confirmed': '✅ 已确认',
    'rejected': '❌ 已驳回',
    'corrected': '✏️ 已修正',
    'escalated': '⬆️ 已升级',
}


def render_review_page(review_workflow: ReviewWorkflow, base_dir: str):
    """渲染复核管理页面"""

    st.markdown('<div class="main-header">📝 复核管理</div>', unsafe_allow_html=True)

    # 顶部：角色选择 + 统计概览
    _render_role_selector()
    _render_stats_overview(review_workflow)

    st.markdown("---")

    # 主体：左侧任务列表，右侧详情
    col_list, col_detail = st.columns([3, 2])

    with col_list:
        _render_task_list(review_workflow)

    with col_detail:
        _render_task_detail(review_workflow)

    # 底部：操作日志
    st.markdown("---")
    _render_action_log(review_workflow)


def _render_role_selector():
    """角色选择器（RBAC）"""
    col1, col2 = st.columns([1, 3])

    with col1:
        current_role = st.session_state.get('review_role', 'operator')
        selected_role = st.selectbox(
            "当前角色",
            options=list(ROLE_LABELS.keys()),
            format_func=lambda k: ROLE_LABELS[k],
            index=list(ROLE_LABELS.keys()).index(current_role)
            if current_role in ROLE_LABELS else 0,
            key="review_role_select",
        )

        if selected_role != current_role:
            st.session_state['review_role'] = selected_role
            st.rerun()

    with col2:
        # 显示当前角色权限
        role = st.session_state.get('review_role', 'operator')
        perms = ROLE_PERMISSIONS.get(role, {})
        st.caption(f"**当前角色：{ROLE_LABELS.get(role, role)}**")

        perm_items = []
        if perms.get('can_view'):
            perm_items.append("👁️ 查看")
        if perms.get('can_create'):
            perm_items.append("➕ 创建")
        if perms.get('can_confirm'):
            perm_items.append("✅ 确认")
        if perms.get('can_reject'):
            perm_items.append("❌ 驳回")
        if perms.get('can_correct'):
            perm_items.append("✏️ 修正")
        if perms.get('can_escalate'):
            perm_items.append("⬆️ 升级")
        if perms.get('can_assign'):
            perm_items.append("📋 分配")

        st.caption("权限：" + " | ".join(perm_items))
        st.caption("【安全提示】权限验证在后端执行，角色仅用于界面演示")


def _render_stats_overview(review_workflow: ReviewWorkflow):
    """统计概览"""
    try:
        stats = review_workflow.get_stats()
    except Exception:
        stats = {
            'total': 0, 'pending': 0, 'confirmed': 0,
            'rejected': 0, 'corrected': 0, 'escalated': 0,
            'today_new': 0, 'annotation_pool_size': 0,
        }

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总任务", stats.get('total', 0))
    col2.metric("⏳ 待处理", stats.get('pending', 0))
    col3.metric("✅ 已确认", stats.get('confirmed', 0))
    col4.metric("❌ 已驳回", stats.get('rejected', 0))
    col5.metric("✏️ 已修正", stats.get('corrected', 0))
    col6.metric("今日新增", stats.get('today_new', 0))


def _render_task_list(review_workflow: ReviewWorkflow):
    """复核任务列表"""
    st.subheader("📋 复核任务列表")

    # 筛选器
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "状态筛选",
            options=["全部", "pending", "confirmed", "rejected", "corrected", "escalated"],
            format_func=lambda x: STATUS_LABELS.get(x, x) if x != "全部" else "全部状态",
            key="review_status_filter",
        )
    with col2:
        limit = st.number_input("显示数量", min_value=10, max_value=200, value=50, step=10)

    # 获取任务列表
    status_param = None if status_filter == "全部" else status_filter
    try:
        tasks = review_workflow.list_tasks(status=status_param, limit=limit)
    except Exception as e:
        st.error(f"加载任务列表失败：{str(e)}")
        tasks = []

    st.metric("任务数量", len(tasks))

    if not tasks:
        st.info("暂无复核任务")
        return

    # 构建显示数据
    display_data = []
    for task in tasks:
        status = task.get('status', 'pending')
        badge_map = {
            'pending': '<span class="quality-badge quality-review">⏳ 待处理</span>',
            'in_progress': '<span class="quality-badge quality-pending">🔄 处理中</span>',
            'confirmed': '<span class="quality-badge quality-confirmed">✅ 已确认</span>',
            'rejected': '<span class="quality-badge quality-rejected">❌ 已驳回</span>',
            'corrected': '<span class="quality-badge quality-corrected">✏️ 已修正</span>',
            'escalated': '<span class="quality-badge quality-fail">⬆️ 已升级</span>',
        }
        badge = badge_map.get(status, status)

        quality = task.get('original_quality', '')
        quality_badge_map = {
            'PASS': '<span class="quality-badge quality-pass" style="font-size:12px;padding:2px 8px;">PASS</span>',
            'REVIEW': '<span class="quality-badge quality-review" style="font-size:12px;padding:2px 8px;">REVIEW</span>',
            'FAIL': '<span class="quality-badge quality-fail" style="font-size:12px;padding:2px 8px;">FAIL</span>',
        }
        quality_badge = quality_badge_map.get(quality, quality)

        display_data.append({
            'ID': task.get('review_id', ''),
            '图片': task.get('image_name', ''),
            '原始判定': quality_badge,
            '状态': badge,
            '优先级': task.get('priority', 'normal'),
            '创建时间': task.get('created_at', '')[:19] if task.get('created_at') else '',
            '处理人': task.get('resolved_by', '') or '-',
        })

    # 使用 HTML 表格显示（带徽章）
    _render_task_table(display_data, tasks)


def _render_task_table(display_data, tasks):
    """渲染任务表格（支持点击选择）"""
    if not display_data:
        return

    # 创建可点击的表格行
    selected_id = st.session_state.get('selected_review_id', '')

    for i, item in enumerate(display_data):
        task = tasks[i]
        review_id = task.get('review_id', '')
        is_selected = review_id == selected_id

        # 根据状态设置行背景
        status = task.get('status', '')
        row_class = ''
        if status == 'pending':
            row_class = 'review-row-pending'
        elif status == 'confirmed':
            row_class = 'review-row-confirmed'
        elif status == 'rejected':
            row_class = 'review-row-rejected'
        elif status == 'corrected':
            row_class = 'review-row-corrected'

        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

        with col1:
            st.markdown(f"**{task.get('image_name', '')}**")
            st.caption(f"ID: {review_id[:20]}...")

        with col2:
            st.markdown(item['原始判定'], unsafe_allow_html=True)

        with col3:
            st.markdown(item['状态'], unsafe_allow_html=True)
            st.caption(f"优先级: {task.get('priority', 'normal')}")

        with col4:
            if st.button("查看详情", key=f"view_{review_id}", use_container_width=True):
                st.session_state['selected_review_id'] = review_id
                st.rerun()

        st.markdown("---")


def _render_task_detail(review_workflow: ReviewWorkflow):
    """任务详情面板"""
    st.subheader("🔍 任务详情")

    review_id = st.session_state.get('selected_review_id', '')

    if not review_id:
        st.info("请从左侧列表选择一个复核任务查看详情")
        return

    try:
        task = review_workflow.get_task(review_id)
    except Exception as e:
        st.error(f"加载任务详情失败：{str(e)}")
        return

    if not task:
        st.warning(f"未找到任务：{review_id}")
        return

    # 基本信息
    st.markdown(f"#### {task.get('image_name', '未知图片')}")
    st.caption(f"任务 ID：{task.get('review_id', '')}")

    # 状态徽章
    status = task.get('status', 'pending')
    badge_map = {
        'pending': '<span class="quality-badge quality-review">⏳ 待处理</span>',
        'in_progress': '<span class="quality-badge quality-pending">🔄 处理中</span>',
        'confirmed': '<span class="quality-badge quality-confirmed">✅ 已确认</span>',
        'rejected': '<span class="quality-badge quality-rejected">❌ 已驳回</span>',
        'corrected': '<span class="quality-badge quality-corrected">✏️ 已修正</span>',
        'escalated': '<span class="quality-badge quality-fail">⬆️ 已升级</span>',
    }
    st.markdown(f"状态：{badge_map.get(status, status)}", unsafe_allow_html=True)

    st.markdown("---")

    # 原始检测和判定信息
    with st.expander("📊 原始检测结果", expanded=True):
        st.write(f"**原始判定：** {task.get('original_quality', '')}")
        st.write(f"**原始评分：** {task.get('original_score', 0)} 分")
        st.write(f"**分析范围：** {task.get('original_scope', '')}")
        st.write(f"**问题描述：** {task.get('reason', '')}")
        st.write(f"**质量问题：** {task.get('quality_issue', '')}")

        # 原始检测数据
        dets_json = task.get('original_detections_json', '[]')
        try:
            dets = json.loads(dets_json) if dets_json else []
        except Exception:
            dets = []

        if dets:
            st.write(f"**缺陷数量：** {len(dets)}")
            with st.expander("查看检测详情"):
                for i, d in enumerate(dets[:20]):
                    if isinstance(d, dict):
                        cls_name = d.get('cls_name', 'unknown')
                        conf = d.get('confidence', 0)
                        st.caption(f"{i+1}. {cls_name} (置信度: {conf:.4f})")
                    elif isinstance(d, (list, tuple)) and len(d) >= 2:
                        st.caption(f"{i+1}. {d[0]} (置信度: {d[1]:.4f})")

        # 原始评估数据
        eval_json = task.get('original_eval_json', '{}')
        try:
            eval_data = json.loads(eval_json) if eval_json else {}
        except Exception:
            eval_data = {}

        if eval_data:
            with st.expander("查看判定详情"):
                st.write(eval_data.get('explanation', '暂无说明'))

    # 版本信息
    with st.expander("📋 版本信息"):
        st.write(f"**模型版本：** {task.get('model_version', 'unknown')}")
        st.write(f"**规则版本：** {task.get('rule_version', 'unknown')}")
        st.write(f"**创建时间：** {task.get('created_at', '')}")
        if task.get('resolved_at'):
            st.write(f"**处理时间：** {task.get('resolved_at', '')}")
            st.write(f"**处理人：** {task.get('resolved_by', '')}")
            st.write(f"**处理结论：** {task.get('resolution', '')}")
            st.write(f"**处理备注：** {task.get('resolution_note', '')}")

    st.markdown("---")

    # 复核操作
    _render_review_actions(review_workflow, task)


def _render_review_actions(review_workflow: ReviewWorkflow, task: dict):
    """复核操作按钮组"""
    st.markdown("#### 复核操作")

    role = st.session_state.get('review_role', 'operator')
    perms = ROLE_PERMISSIONS.get(role, {})
    status = task.get('status', 'pending')
    review_id = task.get('review_id', '')

    # 只有待处理/处理中的任务才能操作
    if status not in ('pending', 'in_progress'):
        st.info("该任务已处理完毕")
        return

    # 备注输入
    note = st.text_area("复核备注", height=80, key=f"note_{review_id}",
                        placeholder="请输入复核意见...")

    col1, col2, col3 = st.columns(3)

    # 确认操作
    with col1:
        if perms.get('can_confirm'):
            if st.button("✅ 确认", use_container_width=True, key=f"confirm_{review_id}"):
                try:
                    review_workflow.confirm(
                        review_id=review_id,
                        actor=f"{role}_user",
                        role=role,
                        note=note,
                    )
                    st.success("✅ 已确认原判定")
                    st.rerun()
                except AuthorizationError as e:
                    st.error(f"权限不足：{str(e)}")
                except Exception as e:
                    st.error(f"操作失败：{str(e)}")
        else:
            st.button("✅ 确认", use_container_width=True, disabled=True,
                     key=f"confirm_{review_id}_disabled")
            st.caption("无确认权限")

    # 驳回操作
    with col2:
        if perms.get('can_reject'):
            corrected_quality = st.selectbox(
                "修正为",
                options=["PASS", "FAIL"],
                format_func=lambda x: "PASS 合格" if x == "PASS" else "FAIL 不合格",
                key=f"reject_quality_{review_id}",
            )
            if st.button("❌ 驳回", use_container_width=True, key=f"reject_{review_id}"):
                if not note.strip():
                    st.warning("请填写驳回原因")
                else:
                    try:
                        review_workflow.reject(
                            review_id=review_id,
                            actor=f"{role}_user",
                            reason=note,
                            corrected_quality=corrected_quality,
                            role=role,
                        )
                        st.success("❌ 已驳回原判定")
                        st.rerun()
                    except AuthorizationError as e:
                        st.error(f"权限不足：{str(e)}")
                    except Exception as e:
                        st.error(f"操作失败：{str(e)}")
        else:
            st.button("❌ 驳回", use_container_width=True, disabled=True,
                     key=f"reject_{review_id}_disabled")
            st.caption("无驳回权限")

    # 升级操作
    with col3:
        if perms.get('can_escalate'):
            if st.button("⬆️ 升级", use_container_width=True, key=f"escalate_{review_id}"):
                if not note.strip():
                    st.warning("请填写升级原因")
                else:
                    try:
                        review_workflow.escalate(
                            review_id=review_id,
                            actor=f"{role}_user",
                            reason=note,
                            role=role,
                        )
                        st.success("⬆️ 已升级处理")
                        st.rerun()
                    except AuthorizationError as e:
                        st.error(f"权限不足：{str(e)}")
                    except Exception as e:
                        st.error(f"操作失败：{str(e)}")
        else:
            st.button("⬆️ 升级", use_container_width=True, disabled=True,
                     key=f"escalate_{review_id}_disabled")
            st.caption("无升级权限")

    # 修正检测结果（高级功能）
    st.markdown("---")
    if perms.get('can_correct'):
        with st.expander("✏️ 修正检测结果（质检员专用）"):
            st.warning(
                "⚠️ 高危操作：修正检测结果将改变质量判定依据。\n\n"
                "此操作仅允许经过培训的质检人员执行。"
                "原始检测结果将保留在记录中，不会被覆盖。"
            )
            corrected_note = st.text_area(
                "修正说明",
                height=60,
                key=f"correct_note_{review_id}",
                placeholder="请说明修正原因和依据..."
            )
            if st.button("提交修正", type="primary", key=f"correct_submit_{review_id}"):
                if not corrected_note.strip():
                    st.warning("请填写修正说明")
                else:
                    try:
                        # 获取原始检测数据作为修正后数据（演示用）
                        dets_json = task.get('original_detections_json', '[]')
                        try:
                            dets = json.loads(dets_json) if dets_json else []
                        except Exception:
                            dets = []

                        eval_json = task.get('original_eval_json', '{}')
                        try:
                            eval_data = json.loads(eval_json) if eval_json else {}
                        except Exception:
                            eval_data = {}

                        review_workflow.correct_detections(
                            review_id=review_id,
                            corrected_detections=dets,
                            corrected_eval=eval_data,
                            actor=f"{role}_user",
                            note=corrected_note,
                            role=role,
                        )
                        st.success("✅ 已提交修正")
                        st.rerun()
                    except AuthorizationError as e:
                        st.error(f"权限不足：{str(e)}")
                    except Exception as e:
                        st.error(f"操作失败：{str(e)}")


def _render_action_log(review_workflow: ReviewWorkflow):
    """操作日志查看"""
    st.subheader("📜 操作日志")

    review_id = st.session_state.get('selected_review_id', '')

    if not review_id:
        st.info("选择一个任务后查看操作日志")
        return

    try:
        actions = review_workflow.get_action_history(review_id)
    except Exception as e:
        st.error(f"加载操作日志失败：{str(e)}")
        return

    if not actions:
        st.caption("暂无操作记录")
        return

    for action in actions:
        action_type = action.get('action', '')
        actor = action.get('actor', '')
        action_time = action.get('action_time', '')
        note = action.get('note', '')

        action_icons = {
            'created': '➕',
            'confirmed': '✅',
            'rejected': '❌',
            'corrected': '✏️',
            'escalated': '⬆️',
        }
        icon = action_icons.get(action_type, '📝')

        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.markdown(f"{icon} **{action_type}**")
        with col2:
            st.write(note)
        with col3:
            st.caption(f"{actor}")
            st.caption(f"{action_time[:19] if action_time else ''}")

        # 显示变更前后数据
        before = action.get('before_data_json')
        after = action.get('after_data_json')
        if before and after and before != '{}':
            with st.expander("查看数据变更"):
                col_b, col_a = st.columns(2)
                with col_b:
                    st.caption("变更前")
                    try:
                        st.json(json.loads(before))
                    except Exception:
                        st.code(before)
                with col_a:
                    st.caption("变更后")
                    try:
                        st.json(json.loads(after))
                    except Exception:
                        st.code(after)
