// Links the names and the venue in the landing hero's author block. DocumenterLandingPage
// renders the hero tagline as plain escaped text, so the links are added here,
// on load; without JavaScript the block reads the same, unlinked.
// AUTHORS: add each author's (and affiliation's) page here once index.md names them.
(function () {
  var LINKS = {
      "CVPR 2027": "https://cvpr.thecvf.com/"
  };
  function linkify() {
    var el = document.querySelector(".landing-tagline");
    if (!el || el.dataset.linked) return;
    el.dataset.linked = "1";
    var text = el.textContent;
    var names = Object.keys(LINKS).sort(function (a, b) { return b.length - a.length; });
    var frag = document.createDocumentFragment(), i = 0;
    while (i < text.length) {
      var hit = null;
      for (var k = 0; k < names.length; k++) {
        if (text.startsWith(names[k], i)) { hit = names[k]; break; }
      }
      if (hit) {
        var a = document.createElement("a");
        a.href = LINKS[hit]; a.textContent = hit;
        frag.appendChild(a); i += hit.length;
      } else {
        var j = i + 1;
        while (j < text.length && !names.some(function (n) { return text.startsWith(n, j); })) j++;
        frag.appendChild(document.createTextNode(text.slice(i, j))); i = j;
      }
    }
    el.textContent = ""; el.appendChild(frag);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", linkify);
  else linkify();
})();
