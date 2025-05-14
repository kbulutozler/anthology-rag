from .retriever import Retriever
import json5
_ret = Retriever()

class PaperSearchTool:
    description = "Search NAACL-2025 papers and return a relevant passage"
    parameters = [{"name":"query","type":"string","description":"search terms","required":True}]
    def call(self, params: str, **_):
        q = json5.loads(params)["query"]
        return json5.dumps({"result":" ".join(p["title"] for p in _ret.search(q))}) 