# -*- coding: utf-8 -*-
"""
人机复核工作流 —— REVIEW 级别的检测结果进入复核队列

功能：
1. 创建复核任务
2. 复核任务列表（按状态/优先级筛选）
3. 处理复核：确认 / 驳回 / 修正检测结果
4. 记录操作历史（谁、什么时间、做了什么、理由）
5. 保留原始证据，不覆盖

安全原则：
- 原始检测结果和初始判定永不修改
- 复核结论与原始结论分开存储
- 困难样本可以进入待标注池，但不得直接用于训练
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class ReviewStatus(str, Enum):
    PENDING = 'pending'        # 待复核
    IN_PROGRESS = 'in_progress'  # 复核中
    CONFIRMED = 'confirmed'    # 已确认（维持原判定）
    REJECTED = 'rejected'      # 已驳回（原判定有误）
    CORRECTED = 'corrected'    # 已修正（修改了检测结果）
    ESCALATED = 'escalated'    # 已升级


class ReviewPriority(str, Enum):
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    URGENT = 'urgent'


# ============================================================
# 角色与权限（后端验证，不相信前端传入的身份）
# ============================================================

# 角色定义及其权限
ROLE_PERMISSIONS = {
    'viewer': {            # 查看者：只能看，不能改
        'can_view': True,
        'can_create': False,
        'can_confirm': False,
        'can_reject': False,
        'can_correct': False,
        'can_escalate': False,
        'can_assign': False,
    },
    'operator': {          # 操作员：可以创建和确认
        'can_view': True,
        'can_create': True,
        'can_confirm': True,
        'can_reject': False,  # 操作员不能直接驳回
        'can_correct': False,  # 不能修正检测结果
        'can_escalate': True,
        'can_assign': False,
    },
    'inspector': {         # 质检员：可以复核和修正
        'can_view': True,
        'can_create': True,
        'can_confirm': True,
        'can_reject': True,
        'can_correct': True,
        'can_escalate': True,
        'can_assign': False,
    },
    'supervisor': {        # 主管：全部权限
        'can_view': True,
        'can_create': True,
        'can_confirm': True,
        'can_reject': True,
        'can_correct': True,
        'can_escalate': True,
        'can_assign': True,
    },
    'system': {            # 系统：创建任务用
        'can_view': True,
        'can_create': True,
        'can_confirm': False,
        'can_reject': False,
        'can_correct': False,
        'can_escalate': False,
        'can_assign': False,
    },
}


class AuthorizationError(Exception):
    """权限不足异常"""
    pass


def _check_permission(role: str, permission: str) -> bool:
    """
    后端权限验证

    【安全原则】权限验证在后端执行，不相信前端传入的身份声明。
    调用方必须提供经过验证的角色（如从会话/JWT中获取），
    不能直接使用用户传入的 actor 字符串。

    Args:
        role: 经过验证的用户角色（viewer/operator/inspector/supervisor/system）
        permission: 要检查的权限名

    Returns:
        bool: 是否有权限

    Raises:
        AuthorizationError: 角色不存在
    """
    if role not in ROLE_PERMISSIONS:
        raise AuthorizationError(f"未知角色: {role}")
    return ROLE_PERMISSIONS[role].get(permission, False)


class ReviewWorkflow:
    """复核工作流管理"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        """获取数据库连接（已设置超时和WAL模式）"""
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout=3000")
        return conn

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")  # 启用 WAL 模式，减少锁竞争
        c = conn.cursor()

        # 复核任务表
        c.execute("""
            CREATE TABLE IF NOT EXISTS review_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT UNIQUE,
                task_id TEXT,
                record_id INTEGER,
                image_name TEXT,
                image_path TEXT,
                original_quality TEXT,
                original_score INTEGER,
                original_scope TEXT,
                quality_issue TEXT,
                reason TEXT,
                priority TEXT DEFAULT 'normal',
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                assigned_to TEXT DEFAULT '',
                resolved_at TEXT,
                resolved_by TEXT,
                resolution TEXT,
                resolution_note TEXT,
                model_version TEXT,
                rule_version TEXT,
                original_detections_json TEXT,
                original_eval_json TEXT
            )
        """)

        # 复核操作历史表
        c.execute("""
            CREATE TABLE IF NOT EXISTS review_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT,
                action TEXT,
                actor TEXT,
                action_time TEXT,
                note TEXT,
                before_data_json TEXT,
                after_data_json TEXT
            )
        """)

        # 待标注池（困难样本）
        c.execute("""
            CREATE TABLE IF NOT EXISTS annotation_pool (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT,
                image_path TEXT,
                image_name TEXT,
                reason TEXT,
                added_at TEXT,
                status TEXT DEFAULT 'pending',
                annotation_json TEXT,
                used_for_training INTEGER DEFAULT 0,
                training_batch TEXT
            )
        """)

        # v2 检测记录表（如果不存在则创建）
        c.execute("""
            CREATE TABLE IF NOT EXISTS detection_records_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                image_name TEXT,
                image_id TEXT,
                timestamp TEXT,
                total_defects INTEGER,
                quality TEXT,
                quality_score INTEGER,
                scope TEXT,
                scope_label TEXT,
                detections_json TEXT,
                eval_json TEXT,
                model_version TEXT,
                rule_version TEXT,
                trace_id TEXT,
                has_review INTEGER DEFAULT 0,
                review_id TEXT DEFAULT ''
            )
        """)

        conn.commit()
        conn.close()

    # ==================== 创建复核任务 ====================

    def create_task(self, task_id: str, record_id: int, image_name: str,
                    image_path: str, original_quality: str, original_score: int,
                    original_scope: str, reason: str, quality_issue: str = '',
                    priority: str = 'normal', model_version: str = '',
                    rule_version: str = '', original_detections: List = None,
                    original_eval: Dict = None) -> str:
        """
        创建复核任务

        Returns:
            review_id: 复核任务ID
        """
        # 使用 image_name + 时间戳生成唯一 ID，避免哈希冲突
        import random
        random_suffix = random.randint(0, 99999)
        review_id = f"rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(image_name + task_id)) % 10000:04d}_{random_suffix:05d}"

        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO review_tasks
                (review_id, task_id, record_id, image_name, image_path,
                 original_quality, original_score, original_scope, quality_issue,
                 reason, priority, status, created_at,
                 model_version, rule_version,
                 original_detections_json, original_eval_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
            """, (
                review_id, task_id, record_id, image_name, image_path,
                original_quality, original_score, original_scope, quality_issue,
                reason, priority, datetime.now().isoformat(),
                model_version, rule_version,
                json.dumps(original_detections or [], ensure_ascii=False),
                json.dumps(original_eval or {}, ensure_ascii=False),
            ))

            # 记录操作
            self._log_action(c, review_id, 'created', 'system',
                            f'创建复核任务，原因: {reason}', None, None)

            conn.commit()
            return review_id
        finally:
            conn.close()

    # ==================== 查询复核任务 ====================

    def list_tasks(self, status: str = None, priority: str = None,
                   limit: int = 50) -> List[Dict]:
        """列出复核任务"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        query = "SELECT * FROM review_tasks WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if priority:
            query += " AND priority = ?"
            params.append(priority)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        c.execute(query, params)
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_task(self, review_id: str) -> Optional[Dict]:
        """获取单个复核任务详情"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM review_tasks WHERE review_id = ?", (review_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_action_history(self, review_id: str) -> List[Dict]:
        """获取复核操作历史"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM review_actions WHERE review_id = ? ORDER BY id",
                 (review_id,))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    # ==================== 处理复核 ====================

    def confirm(self, review_id: str, actor: str = 'operator',
                role: str = 'operator', note: str = '') -> bool:
        """
        确认原判定（维持 REVIEW 或更新为最终结论）

        操作：确认原始检测和判定正确，REVIEW 结论有效

        【安全】需要 can_confirm 权限，角色由后端验证，不相信前端传入的身份。
        actor 参数仅用于审计日志记录。
        """
        # 后端权限验证
        if not _check_permission(role, 'can_confirm'):
            raise AuthorizationError(
                f"角色 '{role}' 无权执行确认操作。"
                f"需要 inspector 或 supervisor 角色（或 operator 仅可确认自己创建的任务）。"
            )

        task = self.get_task(review_id)
        if not task:
            return False

        before = {'status': task['status']}

        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute("""
                UPDATE review_tasks
                SET status = 'confirmed', resolved_at = ?, resolved_by = ?,
                    resolution = 'confirmed', resolution_note = ?
                WHERE review_id = ?
            """, (datetime.now().isoformat(), actor, note, review_id))

            self._log_action(c, review_id, 'confirmed', actor, note,
                            before, {'status': 'confirmed'})

            conn.commit()
            return True
        finally:
            conn.close()

    def reject(self, review_id: str, actor: str = 'operator',
               reason: str = '', corrected_quality: str = 'PASS',
               role: str = 'operator') -> bool:
        """
        驳回原判定（认为判定有误，改为其他等级）

        操作：人工判定产品实际为 PASS/FAIL，原 REVIEW 判定有误

        【安全】需要 can_reject 权限（仅 inspector/supervisor）。
        驳回是高权限操作，操作员不能直接驳回。
        """
        if not _check_permission(role, 'can_reject'):
            raise AuthorizationError(
                f"角色 '{role}' 无权执行驳回操作。"
                f"驳回需要 inspector 或 supervisor 角色。"
            )

        task = self.get_task(review_id)
        if not task:
            return False

        before = {'status': task['status'], 'quality': task['original_quality']}
        after = {'status': 'rejected', 'corrected_quality': corrected_quality}

        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute("""
                UPDATE review_tasks
                SET status = 'rejected', resolved_at = ?, resolved_by = ?,
                    resolution = ?, resolution_note = ?
                WHERE review_id = ?
            """, (datetime.now().isoformat(), actor, corrected_quality, reason, review_id))

            self._log_action(c, review_id, 'rejected', actor, reason, before, after)

            conn.commit()
            return True
        finally:
            conn.close()

    def correct_detections(self, review_id: str, corrected_detections: List,
                            corrected_eval: Dict, actor: str = 'operator',
                            note: str = '', role: str = 'operator') -> bool:
        """
        修正检测结果（人工调整检测框或类别）

        操作：修改检测结果后重新判定
        原始检测结果保留在 original_detections_json 中

        【安全】需要 can_correct 权限（仅 inspector/supervisor）。
        修正检测结果是高危操作，必须由经过培训的质检人员执行。
        """
        if not _check_permission(role, 'can_correct'):
            raise AuthorizationError(
                f"角色 '{role}' 无权修正检测结果。"
                f"修正需要 inspector 或 supervisor 角色。"
            )

        task = self.get_task(review_id)
        if not task:
            return False

        before = {'status': task['status']}
        after = {'status': 'corrected', 'detection_count': len(corrected_detections)}

        conn = self._get_conn()
        try:
            c = conn.cursor()
            c.execute("""
                UPDATE review_tasks
                SET status = 'corrected', resolved_at = ?, resolved_by = ?,
                    resolution = 'corrected', resolution_note = ?
                WHERE review_id = ?
            """, (datetime.now().isoformat(), actor, note, review_id))

            # 修正后的结果存在 action 里，原始数据保留
            self._log_action(c, review_id, 'corrected', actor, note,
                            before, {
                                **after,
                                'corrected_detections': corrected_detections,
                                'corrected_eval': corrected_eval,
                            })

            conn.commit()
            return True
        finally:
            conn.close()

    def escalate(self, review_id: str, actor: str = 'operator',
                 reason: str = '', role: str = 'operator') -> bool:
        """
        升级（提交给更高级别人员处理）

        【安全】需要 can_escalate 权限。operator 及以上角色可升级。
        """
        if not _check_permission(role, 'can_escalate'):
            raise AuthorizationError(
                f"角色 '{role}' 无权执行升级操作。"
            )

        task = self.get_task(review_id)
        if not task:
            return False

        before = {'status': task['status']}

        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""
            UPDATE review_tasks
            SET status = 'escalated', resolution_note = ?
            WHERE review_id = ?
        """, (reason, review_id))

        self._log_action(c, review_id, 'escalated', actor, reason,
                        before, {'status': 'escalated'})

        conn.commit()
        conn.close()
        return True

    # ==================== 待标注池 ====================

    def add_to_annotation_pool(self, review_id: str, reason: str = '') -> bool:
        """
        将困难样本加入待标注池

        重要：加入待标注池不等于可以直接用于训练。
        必须经过人工标注审核后才能使用。
        used_for_training 字段默认为 0。
        """
        task = self.get_task(review_id)
        if not task:
            return False

        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO annotation_pool
            (review_id, image_path, image_name, reason, added_at, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (
            review_id, task['image_path'], task['image_name'],
            reason, datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        return True

    def list_annotation_pool(self, status: str = 'pending', limit: int = 50) -> List[Dict]:
        """列出待标注样本"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT * FROM annotation_pool WHERE status = ?
            ORDER BY id DESC LIMIT ?
        """, (status, limit))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    # ==================== 统计 ====================

    def get_stats(self) -> Dict:
        """获取复核统计"""
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("SELECT status, COUNT(*) FROM review_tasks GROUP BY status")
        status_counts = dict(c.fetchall())

        c.execute("SELECT priority, COUNT(*) FROM review_tasks WHERE status = 'pending' GROUP BY priority")
        pending_by_priority = dict(c.fetchall())

        c.execute("SELECT COUNT(*) FROM review_tasks WHERE date(created_at) = date('now')")
        today_count = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM annotation_pool")
        pool_count = c.fetchone()[0]

        conn.close()

        return {
            'total': sum(status_counts.values()),
            'pending': status_counts.get('pending', 0),
            'in_progress': status_counts.get('in_progress', 0),
            'confirmed': status_counts.get('confirmed', 0),
            'rejected': status_counts.get('rejected', 0),
            'corrected': status_counts.get('corrected', 0),
            'escalated': status_counts.get('escalated', 0),
            'pending_by_priority': pending_by_priority,
            'today_new': today_count,
            'annotation_pool_size': pool_count,
        }

    # ==================== 内部方法 ====================

    def _log_action(self, cursor, review_id: str, action: str, actor: str,
                    note: str, before_data: Dict = None, after_data: Dict = None):
        """记录操作历史"""
        cursor.execute("""
            INSERT INTO review_actions
            (review_id, action, actor, action_time, note,
             before_data_json, after_data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            review_id, action, actor, datetime.now().isoformat(), note,
            json.dumps(before_data or {}, ensure_ascii=False),
            json.dumps(after_data or {}, ensure_ascii=False),
        ))
