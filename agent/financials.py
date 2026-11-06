"""Exact financial numbers from XBRL facts: the agent's get_financial tool.

The model names a company, a metric and a fiscal year; this code chooses the XBRL concept and the filing, so the
number is exact and its source is known. The model never writes SQL. Rules, from looking at the data (README,
Week 4 findings):
  concepts  each metric has an ordered list of candidate concepts, totals before components (`Revenues` before
            `SalesRevenueGoodsNet`, which is goods only); the first one the filing reports wins. Concepts that are
            only a component (`Depreciation` for D&A) are left out rather than returned as the total.
  filing    as reported: the fiscal year's own 10-K first, not a later restatement (3M restated 2022 revenue
            after the Solventum spin-off); only if that 10-K lacks the value, the next 10-Ks, which repeat the
            year as comparison columns. `as_reported=False` takes the latest filing instead (restated values).
  period    flows (revenue, capex) = the full fiscal year (350-380 days) ending on the fiscal year-end;
            balances (inventory, assets) = the value at the fiscal year-end
  forms     10-K facts only, in USD, so proxy-statement data (executive pay) is never returned
"""

import json
import re

import duckdb

from ingest.corpus import CORPUS_PATH
from ingest.xbrl import DB_PATH
from retrieval.filters import ALIASES, normalize

LATER_10KS = 2  # a 10-K repeats the two prior fiscal years, so at most two later 10-Ks can report a year

# metric -> (flow or balance, candidate concepts in priority order, what it means)
METRICS = {
    "revenue": ("flow", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                         "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                         "SalesRevenueGoodsNet"], "total revenue / net sales"),
    "cost_of_revenue": ("flow", ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
                        "cost of revenue / cost of goods sold (COGS)"),
    "gross_profit": ("flow", ["GrossProfit"], "gross profit (gross margin in dollars)"),
    "research_and_development": ("flow", ["ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                                          "ResearchAndDevelopmentExpense"], "research and development expense"),
    "operating_income": ("flow", ["OperatingIncomeLoss"], "operating income (loss)"),
    "net_income": ("flow", ["NetIncomeLoss", "ProfitLoss"],
                   "net income attributable to the company's shareholders"),
    "depreciation_amortization": ("flow", ["DepreciationDepletionAndAmortization",
                                           "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization"],
                                  "depreciation and amortization (cash flow statement)"),
    "operating_cash_flow": ("flow", ["NetCashProvidedByUsedInOperatingActivities"], "cash from operating activities"),
    "capex": ("flow", ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
              "capital expenditures (purchases of property, plant and equipment)"),
    "dividends_paid": ("flow", ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "cash dividends paid"),
    "cash": ("balance", ["CashAndCashEquivalentsAtCarryingValue"], "cash and cash equivalents"),
    "accounts_receivable": ("balance", ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                                        "AccountsNotesAndLoansReceivableNetCurrent"], "accounts receivable, net"),
    "inventory": ("balance", ["InventoryNet", "InventoryFinishedGoodsNetOfReserves",
                              "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings"], "inventories"),
    "current_assets": ("balance", ["AssetsCurrent"], "total current assets"),
    "ppe_net": ("balance", ["PropertyPlantAndEquipmentNet"], "property, plant and equipment, net"),
    "total_assets": ("balance", ["Assets"], "total assets"),
    "accounts_payable": ("balance", ["AccountsPayableCurrent", "AccountsPayableTradeCurrent"], "accounts payable"),
    "current_liabilities": ("balance", ["LiabilitiesCurrent"], "total current liabilities"),
    "total_liabilities": ("balance", ["Liabilities"], "total liabilities"),
    "shareholders_equity": ("balance", ["StockholdersEquity"], "total shareholders' equity (excluding minority)"),
}


class Financials:
    """Read-only access to the XBRL facts in DuckDB, plus the corpus companies (name -> CIK)."""

    def __init__(self):
        self.con = duckdb.connect(str(DB_PATH), read_only=True)
        with open(CORPUS_PATH) as f:
            self.ciks = {r["company"]: r["cik"] for r in map(json.loads, f)}

    def find_company(self, name: str) -> str | None:
        """The corpus company a name refers to ("Apple Inc." -> "Apple", "JnJ" -> "Johnson & Johnson")."""
        text = normalize(name)
        for company in self.ciks:
            if any(normalize(n) in text or text in normalize(n) for n in [company] + ALIASES.get(company, [])):
                return company
        return None

    def ten_ks(self, cik: str) -> dict[int, tuple[str, str]]:
        """Every 10-K of a company: fiscal year -> (accession, fiscal year-end date)."""
        rows = self.con.execute("select fiscal_year, accession, period_end from filings where cik = ? and form = '10-K'",
                                [cik]).fetchall()
        return {fy: (acc, str(end)) for fy, acc, end in rows}

    def get(self, company: str, metric: str, fiscal_year: int, as_reported: bool = True) -> dict:
        """One number with its source, or {"error": ...} saying why there is none."""
        if metric not in METRICS:
            return {"error": f"unknown metric {metric!r}; available: {', '.join(METRICS)}"}
        name = self.find_company(company)
        if name is None:
            return {"error": f"{company!r} is not one of the companies in the corpus"}
        kind, concepts, _ = METRICS[metric]
        cik = self.ciks[name]
        filings = self.ten_ks(cik)
        if fiscal_year not in filings:
            return {"error": f"no 10-K for {name} fiscal {fiscal_year} in the XBRL data"}
        year_end = filings[fiscal_year][1]
        later = [fy for fy in sorted(filings) if fiscal_year < fy <= fiscal_year + LATER_10KS]
        order = [fiscal_year] + later if as_reported else sorted([fiscal_year] + later, reverse=True)
        period = ("period_start is not null and datediff('day', period_start, period_end) between 350 and 380"
                  if kind == "flow" else "period_start is null")
        for fy in order:
            accession = filings[fy][0]
            for concept in concepts:
                rows = self.con.execute(f"""select distinct value, period_start, period_end, filed from facts
                    where cik = ? and accession = ? and concept = ? and unit = 'USD' and form = '10-K'
                      and period_end = ? and {period}""", [cik, accession, concept, year_end]).fetchall()
                if rows:
                    value, start, end, filed = rows[0]
                    return {"company": name, "metric": metric, "fiscal_year": fiscal_year, "value": value,
                            "unit": "USD", "concept": concept, "period_start": str(start) if start else None,
                            "period_end": str(end), "accession": accession, "filed": str(filed),
                            "source": "own 10-K" if fy == fiscal_year else f"fiscal {fy} 10-K",
                            "conflicting_values": [r[0] for r in rows[1:]]}
        return {"error": f"{name} reports no {METRICS[metric][2]} for fiscal {fiscal_year} under the concepts "
                         f"checked ({', '.join(concepts)}); it may be absent or under a company-specific concept",
                "company": name, "metric": metric, "fiscal_year": fiscal_year}


def describe(result: dict) -> str:
    """One line for the LLM or a person: the number in millions, with period and source."""
    if "error" in result:
        return f"Not found: {result['error']}."
    when = (f"{result['period_start']} to {result['period_end']}" if result["period_start"]
            else f"at {result['period_end']}")
    return (f"{result['company']} {result['metric']} fiscal {result['fiscal_year']}: "
            f"${result['value'] / 1e6:,.{0 if result['value'] % 1e6 == 0 else 3}f} million ({when}; XBRL "
            f"{result['concept']}, {result['source']} filed {result['filed']}, accession {result['accession']})")


if __name__ == "__main__":
    fin = Financials()
    for args in [("3M", "revenue", 2022), ("Apple", "research_and_development", 2024),
                 ("Johnson & Johnson", "operating_income", 2022), ("Tesla", "revenue", 2023)]:
        print(describe(fin.get(*args)))
    print(describe(fin.get("3M", "revenue", 2022, as_reported=False)), "<- restated (as_reported=False)")
