"""
Writes data/lab_session/ad_profiles/<ad>.json -- one per ad: the brand (for the
end-of-session brand recall question), the recall answer options, and (for the
SAP / Bosch demo ads) a campaign-objective profile.

Profile rules
  documented  facts from a public source (URL kept with each fact)
  assumed     the research team's reading -- NOT a documented company objective
  public_evidence  campaign-level outcomes measured on OTHER people (not our viewers);
                   never attributed to our EEG.
  objectives  each objective -> how our system tests it (EEG measure + question),
              tagged documented / assumed.

Run once; edit the JSON by hand afterwards if needed (re-running overwrites only
the files listed here; pass --keep to skip files that already exist).
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "data", "lab_session", "ad_profiles")

SRC = {
    "aotw_grow": "https://www.adsoftheworld.com/campaigns/grow-f267fc43-6642-428c-94c0-620c5815f7fc",
    "aotw_justaguy": "https://www.adsoftheworld.com/campaigns/justaguy",
    "adweek_justaguy": "https://www.adweek.com/brand-marketing/bosch-super-bowl-ad-guy-fieri/",
    "adweek_2025": "https://www.adweek.com/creativity/antonio-banderas-macho-man-boschs-big-game-debut/",
    "bsh_2025": "https://www.bsh-group.com/us/en/press/press-releases/bosch-demonstrates-the-more-you-bosch-the-more-you-feel-like-a-bosch-with-first-ever-big-game-ad/",
    "admeter_2026": "https://www.aol.com/articles/best-2026-super-bowl-commercials-144024301.html",
    "kellogg_2026": "https://www.cbsnews.com/news/best-worst-super-bowl-commercials-2026",
    "pollfish_2026": "https://wss.pollfish.com/resources/blog/market-research/the-vampire-effect-the-best-and-worst-super-bowl-2026-commercials",
    "bmi_dw_2020": "https://bestmediainfo.com/2020/08/bosch-india-unveils-first-ever-dishwasher-campaign-busts-common-myths",
    "afaqs_dw_2020": "https://www.afaqs.com/news/advertising/bosch-out-with-6-dishwasher-spots-with-husbands-wives-maids-in-laws",
}

# How each kind of objective is tested in THIS system (same wording everywhere).
TESTS = {
    "brand_linkage": {
        "eeg": "Attention while the brand is on screen vs the rest of the ad (brand-moment test, "
               "label-shuffle p-value, checked on the second viewing)",
        "questions": ["brand_recall (end of session, multiple choice)"],
    },
    "attention_capture": {
        "eeg": "Attention and alertness over the ad, scene by scene, vs the viewer's rest range; "
               "repeat reliability between viewing 1 and 2",
        "questions": ["liking"],
    },
    "message_processing": {
        "eeg": "Processing (frontal theta, exploratory) and attention during the scenes where the "
               "offer is said or shown (transcript + scene labels)",
        "questions": ["purchase_intent / consideration"],
    },
    "liking": {"eeg": "No EEG measure is validated for liking -- self-report only", "questions": ["liking"]},
}

PROFILES = {
    "ad01": {"brand": "Old Spice", "recall_options": ["Old Spice", "Axe", "Dove Men+Care", "Nivea Men"]},
    "ad02": {"brand": "Volkswagen", "recall_options": ["Volkswagen", "Toyota", "Honda", "Ford"]},
    "ad03": {"brand": "Cadbury", "recall_options": ["Cadbury", "Nestle", "Amul", "Hershey's"]},
    "ad04": {"brand": "Budweiser", "recall_options": ["Budweiser", "Heineken", "Coors", "Corona"]},
    "ad05": {"brand": "Google", "recall_options": ["Google", "Bing", "Yahoo", "Microsoft"],
             "recall_note": "The Google search screen is visible through most of the ad, so the still frame "
                            "may show the brand -- treat recall for this ad as a check, not a result."},
    "ad06": {
        "brand": "SAP", "recall_options": ["SAP", "Oracle", "Microsoft", "Salesforce"],
        "title": "GROW", "company": "SAP SE", "year": 2026,
        "profile": {
            "documented": [
                {"field": "Campaign", "value": "GROW (film + integrated), BBDO New York; directed by Rupert Sanders (MJZ)", "source": SRC["aotw_grow"]},
                {"field": "Published", "value": "August 2026, United States", "source": SRC["aotw_grow"]},
                {"field": "Product", "value": "GROW with SAP: AI cloud ERP (finance, supply chain, HR) for mid-sized companies", "source": SRC["aotw_grow"]},
                {"field": "Target audience", "value": "Leaders of ambitious, growing mid-sized companies who may assume SAP is only for global enterprises", "source": SRC["aotw_grow"]},
                {"field": "Core message", "value": "Enterprise-grade technology is for companies with ambition, not just giants; growth shown as visual metaphors (a workspace that stretches, staff added instantly)", "source": SRC["aotw_grow"]},
                {"field": "Channels / markets", "value": "CTV, LinkedIn, YouTube, Google Video; North America, Italy, Germany, Japan", "source": SRC["aotw_grow"]},
                {"field": "SAP quote", "value": "\"people connect with challenges before they connect with solutions\" (Ada Agrait, Global CMO)", "source": SRC["aotw_grow"]},
            ],
            "assumed": [
                {"field": "Intended response", "value": "Mid-size decision makers see SAP as 'for companies like mine' and consider GROW"},
                {"field": "Main risk", "value": "The visual metaphors may hold attention while the actual offer (cloud ERP for mid-size firms) is missed"},
                {"field": "Our viewers", "value": "Engineering students, not mid-size business leaders -- the ad's real audience is not represented"},
            ],
            "public_evidence": [
                {"metric": "Published campaign results", "value": "None found (searched 13 Nov 2026)", "source": None,
                 "note": "No external outcome to compare against."},
            ],
            "objectives": [
                {"objective": "Reposition SAP as relevant for mid-sized companies", "basis": "documented", **TESTS["message_processing"]},
                {"objective": "Hold attention through the visual metaphors", "basis": "assumed", **TESTS["attention_capture"]},
                {"objective": "Link the ad to SAP (brand linkage)", "basis": "assumed", **TESTS["brand_linkage"]},
            ],
        },
    },
    "ad07": {
        "brand": "Bosch", "recall_options": ["Bosch", "IFB", "LG", "Whirlpool"],
        "title": "Indian Kitchen ka Dishwasher (15 s film)", "company": "BSH Home Appliances India (Bosch)", "year": 2020,
        "recall_note": "The Bosch logo is visible from about 10.9 s; the recall still frame is taken from before that.",
        "profile": {
            "documented": [
                {"field": "Campaign", "value": "\"Indian Kitchen ka Dishwasher\" -- Bosch India's first-ever dishwasher campaign, six short digital films; conceptualised by MissMalini Entertainment, directed by Shaun Kolah", "source": SRC["afaqs_dw_2020"]},
                {"field": "Launched", "value": "August 2020; Google Display Network, Facebook, Instagram, then YouTube / Hotstar", "source": SRC["bmi_dw_2020"]},
                {"field": "Objective", "value": "\"address and mitigate myths and assumptions regarding dishwashers\" -- that they don't suit Indian kitchens, utensils, stains or cooking habits", "source": SRC["bmi_dw_2020"]},
                {"field": "Target audience", "value": "Indian households who assume a dishwasher is not made for Indian kitchens", "source": SRC["bmi_dw_2020"]},
                {"field": "Product claims in the campaign", "value": "Water heated up to 70 deg C (99% germ-free), up to 85% less water than hand washing", "source": SRC["bmi_dw_2020"]},
                {"field": "Sign-off line (campaign)", "value": "\"Hope is doubt ko Bosch ne clear kiya\"", "source": SRC["afaqs_dw_2020"]},
                {"field": "One film's theme", "value": "The domestic helper's fear of being made redundant, handled 'with grace and wit' (agency); a critic called it 'flat and superficial'", "source": SRC["afaqs_dw_2020"]},
            ],
            "assumed": [
                {"field": "This film", "value": "A 15 s cut: a house help doubts the machine, the dishwasher is shown with 'Pre-wash. Wash. Dry.', Bosch logo + #IndianKitchenKaDishwasher at the end (from watching the video; which of the six films it is was not confirmed)"},
                {"field": "Intended response", "value": "'A dishwasher does work for Indian utensils' -- doubt removed, consideration raised"},
                {"field": "Main risk", "value": "Brand and product only in the last ~4 s of a 15 s film -- the doubt is shown, the answer is very brief"},
            ],
            "public_evidence": [
                {"metric": "Business context reported at launch (Aug 2020)", "value": "Bosch India: 20% YTD growth vs 2019; dishwasher category demand +250% (MD's statement)", "source": SRC["afaqs_dw_2020"],
                 "note": "Company statements about the market during Covid -- NOT results of this ad, and not our viewers."},
                {"metric": "Published results of this campaign", "value": "None found (searched 13 Nov 2026)", "source": None,
                 "note": "No external outcome to compare against."},
            ],
            "objectives": [
                {"objective": "Remove doubts about whether a dishwasher works for Indian utensils (product understanding)", "basis": "documented", **TESTS["message_processing"]},
                {"objective": "Link the message to Bosch (brand linkage) despite the late logo", "basis": "assumed", **TESTS["brand_linkage"]},
                {"objective": "Hold attention through a 15 s story", "basis": "assumed", **TESTS["attention_capture"]},
            ],
        },
    },
}


def main():
    keep = "--keep" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    for ad, p in PROFILES.items():
        path = os.path.join(OUT, f"{ad}.json")
        if keep and os.path.exists(path):
            continue
        doc = {"ad": ad, **p}
        doc.setdefault("recall_frame_s", None)   # None = picked automatically (longest stretch with no brand on screen)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
        print("wrote", os.path.relpath(path, PROJ))


if __name__ == "__main__":
    main()
