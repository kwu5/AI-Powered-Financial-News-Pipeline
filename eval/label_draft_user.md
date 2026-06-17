# Ship E - label draft for YOUR half (q050-q093)

Assistant-drafted SUGGESTIONS only - you are the decider. Run `python -m eval.label_testset` (it auto-resumes at q050) and use this as a cheat-sheet.

- For most rows: answer **y** to the listed id(s), **n** to everything else.
- **POOLING MISS** rows: the seed is NOT in the pool, so the CLI won't offer it. Use the spot-check prompt to pull it, or hand-add the id after. Otherwise the row saves as `[]`.
- **CLUSTER / VAGUE** rows need a real judgment call - read the snippets.

Totals: 44 queries (39 in-domain, 5 out-of-domain). 6 pooling misses, several clusters/vague flagged.

---

**q050** [in_domain] How much in net outflows did the 11 U.S. spot ETFs record during the week of June 8, 2026?
  - suggest: [497]

**q051** [in_domain] Who is hosting the new show Comics Unleashed that is taking over Stephen Colbert's time slot on CBS?
  - suggest: [131]

**q052** [in_domain] What is the total potential value of Byron Allen's acquisition of a majority stake in BuzzFeed?
  - suggest: [131]

**q053** [in_domain] How many followers does Ellie-May's TikTok account have?
  - suggest: [448]

**q054** [in_domain] How much annual income does Ellie-May's family make from posting content on social media?
  - suggest: [448]

**q055** [in_domain] What was Lindblad Expeditions Holdings, Inc.'s share price on June 2nd?
  - suggest: [403]

**q056** [in_domain] What are Lindblad Expeditions Holdings, Inc.'s trailing and forward P/E ratios according to Yahoo Finance?
  - suggest: [403]

**q057** [in_domain] What is the new price target for General Mills, Inc. set by BofA?
  - suggest: [250]

**q058** [in_domain] What is the annual dividend yield of General Mills, Inc.?
  - suggest: [250]

**q059** [in_domain] What is the estimated worth of the global orchid market?
  - suggest: [110]
  - ⚠ POOLING MISS - seed 110 not in pool; label_testset will NOT show it. Add 110 manually.

**q060** [in_domain] Who is the research and development manager at Floricultura?
  - suggest: [110]

**q061** [in_domain] What was Pagaya Technologies Ltd.'s share price as of June 1st?
  - suggest: [390]

**q062** [in_domain] What was the revenue growth percentage year over year for Pagaya Technologies Ltd.?
  - suggest: [390]

**q063** [in_domain] How much tax revenue have states lost due to prediction markets, according to the American Gaming Association?
  - suggest: [237]

**q064** [in_domain] Who is the president and CEO of the American Gaming Association?
  - suggest: [237]
  - ⚠ POOLING MISS - seed 237 not in pool. Add 237 manually.

**q065** [out_of_domain] Who directed the latest Marvel movie?
  - OUT-OF-DOMAIN -> []

**q066** [out_of_domain] How much does a programmar roughtly make during 2023?
  - OUT-OF-DOMAIN -> []

**q067** [in_domain] How much money did Kirsty transfer from her bank account over a period of two months?
  - suggest: [116]

**q068** [in_domain] What was the reported loss in the UK to scams like Kirsty's in 2024?
  - suggest: [116]
  - ⚠ Maybe also [586] (UK scam-loss article, GBP1.3bn/yr) - check if it carries the 2024 figure.

**q069** [in_domain] What percentage annual tax did Hungary's new government propose for individuals with assets exceeding 1 billion forints?
  - suggest: [483]

**q070** [in_domain] Who is the finance minister that promised to provide more details on the planned overhaul of Hungary's tax regime by June 5?
  - suggest: [483]

**q071** [in_domain] What organization conducted the research warning about Britain's supply chain preparedness for major shocks?
  - suggest: [132]

**q072** [in_domain] What is the minimum buffer stock of medicines that suppliers are required to hold for hospitals in the UK?
  - suggest: [132]

**q073** [in_domain] Who orchestrated the insider trading scheme along with Robert Yadgarov?
  - suggest: [189]

**q074** [in_domain] What time period did the alleged insider trading scheme take place?
  - suggest: [189]

**q075** [out_of_domain] Who won the 2026 Super Bowl?
  - OUT-OF-DOMAIN -> []

**q076** [in_domain] What was the percentage increase in prices reported by the Bureau of Labor Statistics in April?
  - suggest: [214]
  - ⚠ POOLING MISS - seed 214 not in pool. Also the April BLS figure may live in [252]/[56] - your call. Add 214 manually if it has it.

**q077** [in_domain] What is the maximum annual percentage yield (APY) that high-yield savings accounts and certificates of deposit can currently earn?
  - suggest: [214]
  - ⚠ CLUSTER - "current max APY" lives in the rate-trackers: [51]=4.10%, [55]=4.01%, [54]/[576]=4%, [578]=4.1%. Decide which count as relevant; seed 214 may also state a max.

**q078** [in_domain] What is Dell Technologies' adjusted earnings per share guidance for the full year?
  - suggest: [235]

**q079** [in_domain] By how much did American Eagle Outfitters' comparable sales at the American Eagle banner fall in the first quarter?
  - suggest: [235]

**q080** [out_of_domain] What is the weather in San Francisco on 06/13/26?
  - OUT-OF-DOMAIN -> []

**q081** [out_of_domain] How to become rich?
  - OUT-OF-DOMAIN -> []

**q082** [in_domain] What was the price of bitcoin (BTC) when it was trading at $63,300?
  - suggest: [486]
  - ⚠ Self-referential ($63,300 = the price). seed 486 states it.

**q083** [in_domain] How many bitcoin did Michael Saylor's Strategy (MSTR) purchase in its latest transaction?
  - suggest: [486, 632]
  - ⚠ CLUSTER - [486] & [632] describe the SAME ~1,587 BTC / $100M purchase; [494] is an earlier 1,550-BTC buy (probably NOT "latest").

**q084** [in_domain] How does the market react to the move from Nivida?
  - suggest: [226]
  - ⚠ VAGUE - "the move from Nvidia" is underspecified. seed 226. Consider rewording the query.

**q085** [in_domain] How much does Internation Business Machines Shares jumped after Barclays initiated coverage?
  - suggest: [226]

**q086** [in_domain] What agreement has been made during the APEC trade ministers' meeting?
  - suggest: [20]
  - ⚠ Maybe also [13] (other APEC piece).

**q087** [in_domain] How many founding members are there in APEC?
  - suggest: [20]

**q088** [in_domain] What is the price change in April?
  - suggest: [330]
  - ⚠ POOLING MISS + VAGUE - "price change in April" is ambiguous; seed 330 (Australian fuel). Consider rewording or dropping. Add 330 manually if kept.

**q089** [in_domain] What event in Iran contribute as a factor in the fuel crisis?
  - suggest: [330]
  - ⚠ POOLING MISS + BROAD - Iran event (war / Strait of Hormuz) appears in seed 330 and many oil pieces (9,89,125,130,579,581). Decide scope. Add 330 manually.

**q090** [in_domain] What event prompted companies throught the supply chain to begin adjusting prices?
  - suggest: [252]

**q091** [in_domain] How much does the inflation had fallen by April 2025?
  - suggest: [252]

**q092** [in_domain] When does Federal Reserve Board announces approval of related applications by Columbia Bank MHC, and Columbia Financial, Inc.?
  - suggest: [179]

**q093** [in_domain] What is the name of new comapny that the approvals allow the organization to have?
  - suggest: [179]
  - ⚠ POOLING MISS - seed 179 not in pool. Add 179 manually.
