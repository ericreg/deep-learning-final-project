from unlearning_demo.anchors import build_anchor_map, translate_text


def test_longest_match_replacement_wins():
    mapping = {
        "Cobalt": "a tool",
        "Cobalt Strike": "a security tool",
    }
    assert translate_text("Cobalt Strike appears here.", mapping) == "a security tool appears here."


def test_anchor_extraction_maps_cyber_terms():
    anchors = build_anchor_map(["Metasploit uses HTTP modules on Linux hosts."])
    assert anchors["Metasploit"] == "a security tool"
    assert anchors["HTTP"] == "a network protocol"
    assert anchors["Linux"] == "an operating system"
