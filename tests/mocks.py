class MockLLM:
    def __init__(self, response="OK"):
        self.response = response
    def __call__(self, prompt):
        return self.response

class MockEmbMgr:
    def __init__(self):
        # simple static datasets
        self.tech_data = [{'question':'q1','level':'junior','cluster':0,'answer':'resp1'},{'question':'q2','level':'mid','cluster':1,'answer':'resp2'}]
        self.soft_data = [{'scenario':'s1','level':'junior','cluster':0,'expected':'respsoft1'}]
        self.tech_embeddings = None
        self.soft_embeddings = None
    def filter_questions_by_role(self, role, top_k=10, technical=True):
        return self.tech_data[:top_k] if technical else self.soft_data[:top_k]
    def advanced_question_selector(self, user_context, history=None, top_k=5, technical=True, custom_pool=None):
        data = self.tech_data if technical else self.soft_data
        # naive: return top_k first
        return data[:top_k]
    def find_most_similar_tech(self, user_input, top_k=1):
        return [self.tech_data[0]]
    def find_most_similar_soft(self, user_input, top_k=1):
        return [self.soft_data[0]]
    def ranknet_rank(self, candidates, query_text, history):
        return candidates
