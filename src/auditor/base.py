from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AuditStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


_SEVERITY: dict[AuditStatus, int] = {
    AuditStatus.PASS: 0,
    AuditStatus.WARN: 1,
    AuditStatus.FAIL: 2,
}


@dataclass
class Check:
    name: str
    status: AuditStatus
    value: float | str
    threshold: float | str | None
    message: str


@dataclass
class AuditFinding:
    article: str
    checks: list[Check] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    status: AuditStatus = field(init=False)

    def __post_init__(self) -> None:
        self.status = self._worst_status()

    def _worst_status(self) -> AuditStatus:
        if not self.checks:
            return AuditStatus.PASS
        return max(self.checks, key=lambda c: _SEVERITY[c.status]).status
