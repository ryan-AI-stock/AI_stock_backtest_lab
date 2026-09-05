"""Validate preservation policy only; does not certify model or live report readiness."""
import json
from pathlib import Path

EXPECTED_MODELS = {'v4d': 'frozen_formal', 'c6_score0': 'research',
                   'c6_risk': 'challenger', 'actual_holdings': 'forward_tracking'}
REPORTS = {
    '台股加權及中大型權值股訊號追蹤_每日台股報告.pdf',
    '台股加權及中大型權值股訊號追蹤_每週台股報告.pdf',
    '台股股票族群輪動雷達_每日台股報告.pdf',
}
REPOSITORIES = {'AI_stock_market_daily', 'AI_stock_market_weekly', 'AI_stock_rotation_radar',
                'AI_stock_schedule_rules', 'AI_action_orchestrator'}
GATES = {'historical_evidence', 'dependency_review', 'recoverable_version', 'regression_tests'}

def validate(document):
    errors = []
    if document.get('schema_version') != 1:
        errors.append('unsupported schema')
    for name, status in EXPECTED_MODELS.items():
        if document.get('models', {}).get(name, {}).get('status') != status:
            errors.append(f'model boundary changed: {name}')
    for field, required in [('protected_reports', REPORTS), ('protected_repositories', REPOSITORIES),
                            ('legacy_removal_requires', GATES)]:
        if not required.issubset(set(document.get(field, []))):
            errors.append(f'protection missing: {field}')
    if document.get('required_independent_tasks') is not False:
        errors.append('independent tasks must not be required')
    return errors

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'configs/model_lifecycle.json'
    errors = validate(json.loads(path.read_text(encoding='utf-8')))
    print(json.dumps({'policy_valid': not errors, 'errors': errors,
                      'live_readiness_verified': False}, ensure_ascii=False))
    raise SystemExit(bool(errors))
