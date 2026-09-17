import urllib.request
from html.parser import HTMLParser

class DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_a = False
        self.results = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            if "class" in attrs_dict and "result-snippet" in attrs_dict["class"]:
                self.in_a = True
    def handle_data(self, data):
        if self.in_a:
            self.results.append(data)
    def handle_endtag(self, tag):
        if tag == "a":
            self.in_a = False

req = urllib.request.Request("https://html.duckduckgo.com/html/?q=Legal+RAG+architecture+best+practices+chunking+retrieval", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8")
parser = DDGParser()
parser.feed(html)
for i, res in enumerate(parser.results[:5]):
    print(f"{i+1}. {res.strip()}")
