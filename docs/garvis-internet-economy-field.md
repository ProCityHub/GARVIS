# GARVIS Internet Economy Field

GARVIS may research public information, calculate opportunity economics, and prepare proposals for
ProCityHub. Research mode is disabled by default and is enabled only for a run with
`GARVIS_ENABLE_RESEARCH=1`.

## What GARVIS can do

- Read the curated revenue and market learning pack.
- Fetch public pages from an explicit domain allowlist with HTTP GET only.
- Extract source text and same-policy links.
- Rank jobs, contracts, estimating work, quality-control work, digital products, and automation work.
- Calculate hypothetical Bitcoin mining economics.
- Calculate paper position sizing and educational bond prices.
- Produce a source-linked report for Adrien D. Thomas.

## What GARVIS cannot do

- Create or log into accounts.
- Submit applications, bids, forms, messages, or posts.
- Buy hardware or services.
- Place securities or crypto trades.
- Access bank or brokerage accounts.
- Sign a wallet transaction or store private keys, seed phrases, PINs, passwords, or one-time codes.
- Guarantee revenue or investment returns.

## Run one bounded research cycle

```bash
scripts/run_garvis_revenue_research.sh
```

The mission visits no more than twelve source pages and produces recommendations only. Review every
source and calculation before any external action.

## Expand the field

Edit `config/garvis_internet_field.json` and add only domains you have reviewed. The network tool
blocks non-HTTP schemes, credentials in URLs, non-allowlisted domains, private addresses, oversized
responses, and non-text content.
