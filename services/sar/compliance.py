"""
Compliance Validation Module
Validates SAR narratives against FinCEN requirements.
Ensures structural compliance with 5W+H format.
"""
import re


REQUIRED_SECTIONS = [
    'INTRODUCTION', 'WHO', 'WHAT', 'WHEN', 'WHERE',
    'WHY SUSPICIOUS', 'HOW', 'CONCLUSION'
]


def validate_sar(narrative):
    """
    Validate SAR narrative meets FinCEN requirements.
    Returns detailed compliance check results.
    """
    upper = narrative.upper()

    # Section presence
    sections_found = []
    missing_sections = []
    for section in REQUIRED_SECTIONS:
        if section in upper:
            sections_found.append(section)
        else:
            missing_sections.append(section)

    has_all_sections = len(missing_sections) == 0

    # Word count
    word_count = len(narrative.split())
    word_count_ok = 200 <= word_count <= 1000

    # Specific dollar amounts
    has_amounts = bool(re.search(r'\$[\d,]+\.?\d*', narrative))

    # Specific dates
    date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2},? \d{4}'
    has_dates = bool(re.search(date_pattern, narrative))

    # Check for behavioral intelligence references
    behavioral_terms = ['bsi', 'behavioral stability', 'entropy', 'drift', 'burstiness',
                        'counterparty expansion', 'flow velocity', 'structuring score']
    has_behavioral_ref = any(term in narrative.lower() for term in behavioral_terms)

    # Check for counterparty mentions
    has_counterparty = bool(re.search(r'counterpart|beneficiar|recipient|sender', narrative, re.IGNORECASE))

    # Check for risk score mention
    has_risk_score = bool(re.search(r'risk\s*score|risk\s*rating', narrative, re.IGNORECASE))

    # Overall compliance
    compliant = has_all_sections and word_count_ok and has_amounts and has_dates

    return {
        'compliant': compliant,
        'has_all_sections': has_all_sections,
        'sections_found': len(sections_found),
        'sections_total': len(REQUIRED_SECTIONS),
        'missing_sections': missing_sections,
        'word_count': word_count,
        'word_count_ok': word_count_ok,
        'has_specific_amounts': has_amounts,
        'has_specific_dates': has_dates,
        'has_behavioral_reference': has_behavioral_ref,
        'has_counterparty_info': has_counterparty,
        'has_risk_score': has_risk_score,
    }


def format_compliance_report(check):
    """Format compliance check as human-readable report."""
    lines = ["SAR Compliance Validation Report", "=" * 40]

    status = "COMPLIANT" if check['compliant'] else "REVIEW REQUIRED"
    lines.append(f"Status: {status}")
    lines.append(f"Sections: {check['sections_found']}/{check['sections_total']}")

    if check['missing_sections']:
        lines.append(f"Missing: {', '.join(check['missing_sections'])}")

    lines.append(f"Word Count: {check['word_count']} ({'OK' if check['word_count_ok'] else 'FAIL'})")
    lines.append(f"Dollar Amounts: {'PASS' if check['has_specific_amounts'] else 'FAIL'}")
    lines.append(f"Specific Dates: {'PASS' if check['has_specific_dates'] else 'FAIL'}")
    lines.append(f"Behavioral Intelligence: {'PASS' if check['has_behavioral_reference'] else 'MISSING'}")
    lines.append(f"Counterparty Info: {'PASS' if check['has_counterparty_info'] else 'MISSING'}")
    lines.append(f"Risk Score Referenced: {'PASS' if check['has_risk_score'] else 'MISSING'}")

    return "\n".join(lines)
