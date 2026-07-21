"""Curated economic learning pack for GARVIS and ProCityHub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RevenueLane:
    name: str
    description: str
    deliverables: tuple[str, ...]
    proof_required: tuple[str, ...]
    execution_boundary: str


REVENUE_LANES: tuple[RevenueLane, ...] = (
    RevenueLane(
        name="Drywall estimating and bid review",
        description=(
            "Use Adrien D. Thomas's trade experience to prepare drywall takeoffs, scope reviews, "
            "bid comparisons, exclusions, and estimate drafts."
        ),
        deliverables=(
            "drawing and specification summary",
            "quantity and scope worksheet",
            "assumptions and exclusions register",
            "human-reviewed bid package",
        ),
        proof_required=("source drawings", "scope documents", "human trade review"),
        execution_boundary="Adrien approves price, bid, and delivery.",
    ),
    RevenueLane(
        name="Drywall quality-control reports",
        description=(
            "Organize site photos and notes into deficiency reports, completion lists, repair plans, "
            "and builder handover documentation."
        ),
        deliverables=(
            "evidence-indexed deficiency report",
            "priority and responsibility matrix",
            "repair estimate draft",
            "closeout checklist",
        ),
        proof_required=("dated photos", "location labels", "applicable specifications"),
        execution_boundary="Adrien verifies technical findings before release.",
    ),
    RevenueLane(
        name="Construction workflow automation",
        description=(
            "Design read-only and approval-gated systems for intake, estimating, deficiency tracking, "
            "daily reports, change orders, and follow-up."
        ),
        deliverables=("workflow map", "prototype", "audit log", "deployment checklist"),
        proof_required=("client requirements", "test cases", "approval boundaries"),
        execution_boundary="No production deployment or account change without approval.",
    ),
    RevenueLane(
        name="Manuals, SOPs, and training",
        description=(
            "Turn verified trade knowledge into inspection manuals, onboarding guides, checklists, "
            "and instructional material."
        ),
        deliverables=("manual", "checklists", "training outline", "revision log"),
        proof_required=("trade review", "source standards", "version history"),
        execution_boundary="Adrien approves technical claims and publication.",
    ),
    RevenueLane(
        name="Tender and lead research",
        description=(
            "Find legitimate projects, employers, contractors, and service opportunities; verify each "
            "source and rank it by expected value and risk."
        ),
        deliverables=("source-linked opportunity brief", "risk screen", "draft outreach"),
        proof_required=("original posting", "verified organization", "closing date"),
        execution_boundary="No outreach, application, or submission without approval.",
    ),
    RevenueLane(
        name="Digital construction products",
        description=(
            "Create reusable estimating worksheets, inspection templates, deficiency forms, scope "
            "templates, and educational products."
        ),
        deliverables=("product file", "instructions", "quality review", "change log"),
        proof_required=("user need", "accuracy review", "license and source review"),
        execution_boundary="No storefront listing or sale without approval.",
    ),
    RevenueLane(
        name="Job matching and placement research",
        description=(
            "Compare verified jobs with worker skills, wage requirements, distance, schedule, and "
            "qualifications; prepare application material for human review."
        ),
        deliverables=("ranked job list", "match explanation", "resume draft", "cover-letter draft"),
        proof_required=("original posting", "employer verification", "candidate consent"),
        execution_boundary="No application or representation of a worker without approval.",
    ),
    RevenueLane(
        name="Market and bond education laboratory",
        description=(
            "Study securities, yields, risk, diversification, and portfolio mathematics using public "
            "data and paper positions only."
        ),
        deliverables=("market brief", "paper portfolio", "risk scenarios", "learning log"),
        proof_required=("timestamped source", "calculation record", "risk disclosure"),
        execution_boundary="No live orders, brokerage login, or financial transfers.",
    ),
    RevenueLane(
        name="Bitcoin proof-of-work laboratory",
        description=(
            "Study mining economics, block-header hashing, difficulty, hashrate, energy cost, and pool "
            "mechanics through simulation and read-only monitoring."
        ),
        deliverables=("profitability estimate", "simulation result", "hardware feasibility report"),
        proof_required=("network inputs", "power price", "hardware assumptions"),
        execution_boundary="No wallet signing, seed phrase, mining account, or purchase without approval.",
    ),
)

OFFICIAL_SOURCE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "employment": ("jobbank.gc.ca", "canada.ca"),
    "tax": ("canada.ca",),
    "bonds_and_rates": ("bankofcanada.ca",),
    "investor_protection": ("ciro.ca", "securities-administrators.ca"),
    "bitcoin_protocol": ("developer.bitcoin.org", "bitcoin.org"),
    "software": ("github.com", "raw.githubusercontent.com"),
}

MARKET_RULES: tuple[str, ...] = (
    "Research and paper simulation are allowed; live trading is disabled.",
    "Never request or store a wallet seed phrase, private key, bank password, PIN, or one-time code.",
    "Never describe an investment return as guaranteed.",
    "Separate historical data, current source data, assumptions, and forecasts.",
    "Every recommendation must include downside risk, liquidity risk, and evidence quality.",
    "GARVIS prepares proposals; Adrien D. Thomas controls every external action.",
)


def learning_context(topic: str = "") -> str:
    """Return a compact, deterministic learning context for the requested topic."""

    normalized = topic.casefold().strip()
    selected = [lane for lane in REVENUE_LANES if not normalized or normalized in lane.name.casefold()]
    if not selected:
        selected = list(REVENUE_LANES)

    sections: list[str] = []
    for lane in selected:
        sections.append(
            "\n".join(
                (
                    f"LANE: {lane.name}",
                    f"PURPOSE: {lane.description}",
                    f"DELIVERABLES: {', '.join(lane.deliverables)}",
                    f"PROOF: {', '.join(lane.proof_required)}",
                    f"BOUNDARY: {lane.execution_boundary}",
                )
            )
        )
    sections.append("MARKET RULES:\n- " + "\n- ".join(MARKET_RULES))
    return "\n\n".join(sections)
