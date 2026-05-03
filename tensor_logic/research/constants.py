from .utils import Schema

FAMILY = Schema("family", {
    "parent": ("person", "person"),
    "sibling": ("person", "person"),
})

WORKPLACE = Schema("workplace", {
    "manages": ("person", "person"),
    "peer_of": ("person", "person"),
})

CONTACTS = Schema("contacts", {
    "knows":      ("person", "person"),
    "manages":    ("person", "person"),
    "peers_with": ("person", "person"),
})

DISTRACTORS = {
    "lives_in": ("person", "city"),
    "born_in": ("person", "city"),
    "owns": ("person", "thing"),
    "knows_lang": ("person", "lang"),
    "speaks": ("person", "lang"),
    "likes":      ("person", "person"),
    "trusts":     ("person", "person"),
    "reports_to": ("person", "person"),
    "envies":     ("person", "person"),
    "admires":    ("person", "person"),
}

def schema_with_distractors(schema: Schema) -> Schema:
    rels = dict(schema.relations)
    rels.update(DISTRACTORS)
    return Schema(schema.name + "+distract", rels)

GOLD_RULES = {
    "grandparent": (FAMILY, ["parent", "parent"]),
    "uncle":       (FAMILY, ["sibling", "parent"]),
    "great_uncle": (FAMILY, ["sibling", "parent", "parent"]),
    "skip_manager":     (WORKPLACE, ["manages", "manages"]),
    "skip_peer":        (WORKPLACE, ["peer_of", "manages"]),
    "skip_skip_manager":(WORKPLACE, ["manages", "manages", "manages"]),
}

QUERY_TARGETS = [
    ("friend_of_friend",    ["knows", "knows"]),
    ("skip_manager",        ["manages", "manages"]),
    ("managed_peer",        ["manages^T", "manages"]),
    ("peer_friend",         ["peers_with", "knows"]),
    ("friend_peer",         ["knows", "peers_with"]),
    ("managed_friend",      ["manages^T", "knows"]),
    ("skip_skip_manager",   ["manages", "manages", "manages"]),
    ("friend_skip_manager", ["knows", "manages", "manages"]),
    ("peer_skip_manager",   ["peers_with", "manages", "manages"]),
    ("managed_peer_friend", ["manages^T", "peers_with", "knows"]),
]
