from collections import defaultdict

class TokenIndex:
    def __init__(self):
        # Maps country -> token -> set of entity_ids
        self.index = defaultdict(lambda: defaultdict(set))
        self.entity_store = {}
        
    def add_record(self, record):
        country = record.get("country", "")
        eid = record.get("entity_id")
        if not eid:
            return
            
        tokens = record.get("name_tokens", set())
        self.entity_store[eid] = record
        
        # Only index tokens that are non-empty
        for token in tokens:
            if token:
                self.index[country][token].add(eid)
            
    def search(self, country, tokens, min_overlap=1):
        if not country or country not in self.index:
            return set()
            
        candidate_counts = defaultdict(int)
        for token in tokens:
            if token and token in self.index[country]:
                for eid in self.index[country][token]:
                    candidate_counts[eid] += 1
                    
        return {eid for eid, count in candidate_counts.items() if count >= min_overlap}
