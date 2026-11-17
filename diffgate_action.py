"""GitHub Action entry for styxx.diffgate — checkout-free, injection-safe.

Reads the event payload from GITHUB_EVENT_PATH (the summary text never touches a shell),
fetches the diff from the GitHub API with the workflow token, gates the summary against it
with ``styxx.diffgate.gate_diff_text``, writes a job-summary table, emits ::error::
annotations for contradictions, and exits per the gate verdict and the strict/soft-fail
inputs. Two events are gated: ``pull_request`` (the description against the pull request's
diff) and ``push`` (the head commit's message against the compare diff); a push that creates a
tag is not gated. On every other event (``pull_request_target`` and ``merge_group`` among them)
the script prints one line saying the event is not gated, reads no claim, writes no summary and
exits 0, whatever strict and soft-fail are set to.

The text is read as it was written. A description is read in its Markdown source, so an HTML
comment, a task-list item and a sentence inside a code fence are read like any other sentence.

The gate that runs is the ``styxx`` package beside this file. ``action.yml`` starts the script by
its path, Python then looks in the script's own folder before any installed package, and the
import below finds ``styxx`` in the Action's checkout, at the ref the workflow names in ``uses:``.
What ``action.yml``'s styxx-version input installs supplies the one package that code imports from
outside the standard library (numpy); it does not choose the gate.

What is printed for the claims: a row of the summary's table for each, up to 524,288 bytes of rows
(a line under the table counts the claims past that by verdict); an ``::error`` command for every
contradicted claim, under strict for every UNCHECKABLE one, and in a blocking check for every claim
whose CONTRADICTED reading the overlay withheld; and a ``::warning`` command for at most 100 of the
other claims the gate could not check, with one more that counts the rest where there are more.
GitHub's documentation of its checks API says "GitHub Actions are limited to 10 warning annotations
and 10 error annotations per step", so a claim past the tenth of its kind in the step has its row
in the table and its line in the job's log, and may have no annotation.

soft-fail (STYXX_SOFT_FAIL, which action.yml fills from its soft-fail input) is read by
``_soft_fail``, and only an explicit "false" blocks. The value is compared case-insensitively
with surrounding whitespace stripped. "false" blocks: the job fails on a contradicted claim; on a
claim the gate reports UNCHECKABLE because its PATH-2a overlay withheld a CONTRADICTED reading of it;
and on a diff that could not be fetched, or that the gate did not measure, while the text holds a
claim the gate reads. With strict on it fails on every UNCHECKABLE claim too, and on a diff it did not
read whatever a non-empty text holds (an empty one is not gated). "true" reports: every verdict goes to the job summary and
the annotations. Unset reads as "true", the input's default. Every other value, the empty string
included, reports as "true" does, and the script prints a warning naming the value it did not
recognise. In report mode the gate's verdicts never fail the job; an error the script does not
catch (an unreadable event file, for one) still does.
The reason the default reports, with its receipts, is in action.yml's description and the
CHANGELOG entry "the GitHub Action reports by default".

The workflow token (GH_TOKEN, which action.yml fills from github.token) goes in a request addressed
to one origin and to no other: the origin of GITHUB_API_URL, which the runner sets
(https://api.github.com on github.com; a GitHub Enterprise Server names its own), or that of
https://api.github.com where the variable is unset or empty. ``api`` asks nothing of an address off
that origin, follows a redirect only within it, and refuses any other; a refusal is a fetch that
failed, so the gate reports that it did not run. Where the environment names a proxy, urllib hands
it the request as it does any other, so a proxy in front of a plain-http origin reads the token.
The CHANGELOG entry "the GitHub Action sends the workflow token in a request addressed to the API's
origin and to no other" says what was sent where before.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from styxx.diffgate import gate_diff_text

# PATH-2a (NOTE_path2a_third_pass_2026_09_30): the form of a reason the overlay writes, at its start, and the kinds
# it may move. Written out here, not imported, so this script reads a styxx without the overlay too
# (tests/test_diffgate_path2a.py pins the kinds to the module's REACH). Run as the Action runs it,
# `python <action path>/diffgate_action.py`, it imports the styxx package beside it, at the ref the workflow names
# (NOTE_path2a_sixth_pass_2026_09_30, I-1), not the one pip installs.
_OVERLAY_WHY = re.compile(r"(?:VERIFIED|CONTRADICTED) withheld by PATH-2a \((?:#97|#121|#97, #121|#101)\): ")
_OVERLAY_MAIN = ". main's reading: "
_OVERLAY_KINDS = frozenset({"file_created", "file_deleted", "file_touched", "files_changed_count", "only_touches",
                            "tests_added", "symbol_added"})


_GITHUB_API = "https://api.github.com"      # the API on github.com: the origin where the runner names none
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_DEFAULT_PORT = {"https": 443, "http": 80}
# The only shape of host and port this script reads: a name of ASCII letters, digits, ".", "-", "_" and "~", or an
# IPv6 literal in its brackets, then at most a colon and ASCII digits. No user part, no "%", and nothing beside
# the brackets but the port, so what urlsplit calls the host is the whole of it on every Python.
_NETLOC = re.compile(r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._~-]+)(?::[0-9]*)?")


def _origin(url):
    """(scheme, host, port) of an address, the port filled in from the scheme. None for a value that
    is not a string, for an address holding anything but printable ASCII (a space, a control
    character, a character past ASCII) anywhere, and for one whose host and port are not in the one
    shape ``_NETLOC`` allows (a user part, a "%", a port that is not ASCII digits, text beside a
    bracketed host). ``styxx.diffgate._origin`` makes the same reading of every address this one
    reads; this one reads fewer, so that where a request goes does not hang on how lenient the
    running Python's urlsplit is (tests/test_diffgate_action_token_redirect.py holds the two side by
    side)."""
    if not isinstance(url, str) or any(c <= " " or c >= "\x7f" for c in url):
        return None
    try:
        p = urllib.parse.urlsplit(url)
        port = p.port
    except ValueError:
        return None
    if not p.hostname or _NETLOC.fullmatch(p.netloc) is None:
        return None
    scheme = p.scheme.lower()
    return (scheme, p.hostname.lower(), port if port is not None else _DEFAULT_PORT.get(scheme))


def _api_origin():
    """The one origin the workflow token is sent to: the origin of GITHUB_API_URL, or that of
    https://api.github.com where the variable is unset or empty. None where the variable holds
    something that is not an http or https address this script can read; the token is then sent
    nowhere."""
    origin = _origin((os.environ.get("GITHUB_API_URL") or "").strip() or _GITHUB_API)
    return origin if origin is not None and origin[0] in _DEFAULT_PORT else None


def _origin_text(origin) -> str:
    scheme, host, port = origin
    if ":" in host:                         # an IPv6 literal goes back in its brackets
        host = f"[{host}]"
    return f"{scheme}://{host}" if port == _DEFAULT_PORT[scheme] else f"{scheme}://{host}:{port}"


def _on_origin(url, origin):
    """``url`` written out again on ``origin``: the scheme, host and port as ``_origin_text`` writes
    them, then the path and the query as given (a missing path is written "/"; a "?" with nothing
    after it and a fragment are left out). None where ``url`` is not on that origin. urllib, and a
    proxy where the environment names one, are handed this text and never the text that was checked,
    so no second reading of an oddly written host or port can part from the check."""
    if origin is None or _origin(url) != origin:
        return None
    p = urllib.parse.urlsplit(url)
    return _origin_text(origin) + (p.path or "/") + (f"?{p.query}" if p.query else "")


def _hide(text: str, token) -> str:
    """``text`` with the token's value replaced wherever it stands, as it was set and as a server
    reads it from the header (without the spaces round it). A token that itself holds ``***`` is
    outside what this can promise; no token GitHub issues holds an asterisk."""
    token = token or ""
    for value in dict.fromkeys((token, token.strip())):
        if value:
            text = text.replace(value, "***")
    return text


def _shown(value, token: str = "") -> str:
    """A value as a refusal names it: quoted with ``repr``, which keeps a line break out of it; cut
    at 200 characters before the quoting and at 400 after it, since ``repr`` can write one character
    as ten; and with the token's value replaced where the value carries it, before the cut and again
    in what ``repr`` wrote, because an escape ``repr`` writes can end in the character a token
    starts with."""
    if not isinstance(value, str):
        return repr(value)[:200]
    shown = _hide(repr(_hide(value, token)[:200]), token)
    return shown if len(shown) <= 400 else shown[:399] + shown[0]


class _ApiRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to the API's origin (GitHub answers a renamed or moved repository
    with one, to another path there), and puts the token back on that one only. A redirect anywhere
    else is not followed: urllib raises it as an HTTPError and ``api`` says so. The token travels as
    an unredirected header, which urllib's own redirect handling never copies to a redirected
    request; this handler is the one place in the script it is copied, and only within the origin.
    The request it follows with is addressed in the script's own spelling (``_on_origin``). 308 is
    followed on every Python: urllib's handler follows it only from Python 3.11, and for a GET a 308
    asks exactly what a 307 asks.

    It also keeps, for ``api``'s sentence, what the last redirecting answer was and what became of
    it: ``answer`` is (code, the address it named or None), read as urllib reads it (Location, else
    URI; an empty value is a value); ``verdict`` is "refused" where this handler turned it down,
    "followed" where it handed urllib a request, and None where urllib stopped before asking."""

    def __init__(self, origin):
        self._api_origin = origin
        self.answer = None
        self.verdict = None

    def http_error_302(self, req, fp, code, msg, headers):
        named = headers.get("Location")
        if named is None:
            named = headers.get("URI")
        self.answer, self.verdict = (code, named), None
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = _on_origin(newurl, self._api_origin)
        if target is None:
            self.verdict = "refused"
            return None
        new = super().redirect_request(req, fp, 307 if code == 308 else code, msg, headers, target)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth is not None:
            new.add_unredirected_header("Authorization", auth)
        self.verdict = "followed"
        return new


def _token() -> str:
    """GH_TOKEN as it will be sent. A value that is unset, or that holds anything but printable ASCII,
    is refused here in a sentence that does not quote it. Of those, http.client would refuse a bare
    line break in a sentence that quotes the value and a character past Latin-1 in one that names the
    character and its place; any other control character (a tab round the token among them) and a
    Latin-1 character it would send."""
    token = os.environ.get("GH_TOKEN")
    if token is None:
        raise RuntimeError("GH_TOKEN is not set, so the diff was not asked for")
    if any(c < " " or c >= "\x7f" for c in token):
        raise RuntimeError("GH_TOKEN holds a line break, another control character or a character "
                           "past ASCII, so it was sent nowhere and the diff was not asked for")
    return token


def api(url: str, accept: str) -> str:
    """The body GitHub's API serves at ``url``, asked with the workflow token.

    The request, token included, is addressed to the API's origin (``_api_origin``) and to no other;
    where the environment names a proxy, urllib hands the request to it as it does any other, so for
    a plain-http origin the proxy reads the token, and for an https origin it sees the host and port
    and no more. An address off the origin is not asked at all. The request goes through an opener of
    this script's own (one installed with ``urllib.request.install_opener`` is not used), which
    follows a redirect only within the origin and refuses any other. Each refusal is raised as a
    RuntimeError of one sentence, in which an address is cut at 200 characters and quoted with
    ``repr``, and the token's value is replaced wherever an address a server named carries it back.
    Where more than one applies, the origin's comes before the address's and the token's last."""
    origin = _api_origin()
    if origin is None:
        raise RuntimeError(f"GITHUB_API_URL is {_shown(os.environ.get('GITHUB_API_URL') or '')}, which is "
                           "not an http or https address this script can read, so the workflow token "
                           "is sent nowhere and the diff was not asked for")
    base = _origin_text(origin)
    read = _origin(url)
    if read is None or read[0] not in _DEFAULT_PORT:    # no scheme, or another: the same sentence on every Python
        raise RuntimeError(f"the diff's address {_shown(url)} is not one this script can read (it "
                           "reads an http or https address in printable ASCII with a plain host and "
                           "port), so it was not asked for")
    target = _on_origin(url, origin)
    if target is None:
        raise RuntimeError(f"the diff's address {_shown(url)} is not on {base}, the one origin the "
                           "workflow token is sent to, so it was not asked for")
    token = _token()
    req = urllib.request.Request(target, headers={
        "Accept": accept, "User-Agent": "styxx-diffgate-action",
        "X-GitHub-Api-Version": "2022-11-28"})
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    handler = _ApiRedirect(origin)
    opener = urllib.request.build_opener(handler)
    try:
        with opener.open(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code not in _REDIRECT_CODES:
            raise
        named = handler.answer[1] if handler.answer else None
        shown = _shown(named, token) if named is not None else "(none given)"
        if handler.verdict == "followed":   # urllib stops a loop after this handler has agreed to the hop
            why = (f"within {base}, which was not followed: the redirects loop or are more than "
                   "urllib follows")
        else:
            why = (f"which was not followed: only a redirect within {base} is, and the workflow "
                   "token is sent nowhere else")
        raise RuntimeError(f"GitHub answers {e.code}, a redirect to {shown}, {why}") from e
    except ValueError as e:
        # urllib reads the address a redirect names before it asks the handler, and raises a ValueError
        # for one it cannot read, in words of its own: "Invalid IPv6 URL" for an unbalanced bracket, and
        # for other bracketed text a sentence that quotes what stands in the brackets, uncut.
        if handler.answer is None or handler.verdict is not None:
            raise
        code, named = handler.answer
        raise RuntimeError(f"GitHub answers {code}, a redirect to {_shown(named, token)}, which urllib "
                           "could not read as an address, so it was not followed") from e


# Characters that are written as escapes wherever this script prints a claim, a reason or the text of
# a failed fetch, beside the control characters. By the Unicode standard's account they have no glyph
# of their own, and some change the order the text round them is shown in, so a reader of the summary
# or of an annotation may not see what the text holds. The list is the standard's default ignorable
# code points (Unicode 15.0) and the three interlinear annotation characters, without the ones the
# standard gives the drawing of a neighbour to: the two joiners (U+200C, U+200D) and the variation
# selectors (U+180B to U+180D, U+180F, U+FE00 to U+FE0F, U+E0100 to U+E01EF). It is written out here
# and not asked of ``unicodedata``, whose answer moves with the Python that runs the script.
_UNSEEN = frozenset(chr(c) for low, high in (
    (0x00AD, 0x00AD),        # soft hyphen
    (0x034F, 0x034F),        # combining grapheme joiner
    (0x061C, 0x061C),        # Arabic letter mark
    (0x115F, 0x1160),        # Hangul fillers
    (0x17B4, 0x17B5),        # Khmer inherent vowels
    (0x180E, 0x180E),        # Mongolian vowel separator
    (0x200B, 0x200B),        # zero width space
    (0x200E, 0x200F),        # left-to-right and right-to-left marks
    (0x202A, 0x202E),        # directional embeddings and overrides, and the pop that ends them
    (0x2060, 0x206F),        # word joiner, invisible operators, directional isolates, deprecated format characters, U+2065
    (0x3164, 0x3164),        # Hangul filler
    (0xFEFF, 0xFEFF),        # zero width no-break space (a byte order mark)
    (0xFFA0, 0xFFA0),        # halfwidth Hangul filler
    (0xFFF0, 0xFFFB),        # nine code points not assigned, and interlinear annotation
    (0x1BCA0, 0x1BCA3),      # shorthand format controls
    (0x1D173, 0x1D17A),      # musical format controls
    (0xE0000, 0xE00FF),      # the tag characters, and the code points round them that are not assigned
    (0xE01F0, 0xE0FFF),      # the rest of that plane's default ignorable code points, none assigned
) for c in range(low, high + 1))


# The two joiners and the variation selectors, which the list above leaves out because by the
# standard's account the character beside them can be drawn by them. Whether one is written as itself
# turns on the characters beside it, and not on whether the standard lists the pair: one of the 258
# selectors other than U+FE0E and U+FE0F after any character past U+009F that is written as itself
# and is no joiner or selector; U+FE0E and U+FE0F after any character of ``_EMOJI`` below, or in a
# keycap; a joiner between two such characters, or between a selector written as itself and one
# such character (``_printable`` says it in full). By the standard's account many of those places
# have nothing to draw, U+FE00 after an accented letter among them. Elsewhere they are written as
# escapes: between two ASCII letters of a path they have nothing to draw, and by the standard's
# account they have no glyph there.
_JOINERS = frozenset("\u200c\u200d")
_SELECTORS = frozenset(chr(c) for low, high in (
    (0x180B, 0x180D), (0x180F, 0x180F),   # Mongolian free variation selectors
    (0xFE00, 0xFE0F),                     # variation selectors 1 to 16
    (0xE0100, 0xE01EF),                   # variation selectors 17 to 256
) for c in range(low, high + 1))
_KEYCAP_BASES = frozenset("#*0123456789")
# U+FE0E and U+FE0F choose between the text form and the emoji form of the character before them. The
# standard gives them that work only after a character with its Emoji property, and of those only
# after the ones emoji-variation-sequences.txt lists (UTS #51, ED-8a and ED-9a); in Unicode 15.0 that
# file lists 342 past U+009F. They are written as themselves after any character of the table below,
# or in a keycap. So after 1,070 of its characters, U+1F600 among them, they are written as
# themselves where the standard lists no sequence. The table holds the characters past U+009F with
# the Emoji property in Unicode 15.0, written out here: 1,412 code points.
_STYLE_SELECTORS = frozenset("\ufe0e\ufe0f")
_EMOJI = frozenset(chr(c) for item in (
    0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, (0x2194, 0x2199), (0x21A9, 0x21AA),
    (0x231A, 0x231B), 0x2328, 0x23CF, (0x23E9, 0x23F3), (0x23F8, 0x23FA), 0x24C2, (0x25AA, 0x25AB),
    0x25B6, 0x25C0, (0x25FB, 0x25FE), (0x2600, 0x2604), 0x260E, 0x2611, (0x2614, 0x2615), 0x2618,
    0x261D, 0x2620, (0x2622, 0x2623), 0x2626, 0x262A, (0x262E, 0x262F), (0x2638, 0x263A), 0x2640,
    0x2642, (0x2648, 0x2653), (0x265F, 0x2660), 0x2663, (0x2665, 0x2666), 0x2668, 0x267B,
    (0x267E, 0x267F), (0x2692, 0x2697), 0x2699, (0x269B, 0x269C), (0x26A0, 0x26A1), 0x26A7,
    (0x26AA, 0x26AB), (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5), 0x26C8,
    (0x26CE, 0x26CF), 0x26D1, (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5),
    (0x26F7, 0x26FA), 0x26FD, 0x2702, 0x2705, (0x2708, 0x270D), 0x270F, 0x2712, 0x2714, 0x2716,
    0x271D, 0x2721, 0x2728, (0x2733, 0x2734), 0x2744, 0x2747, 0x274C, 0x274E, (0x2753, 0x2755),
    0x2757, (0x2763, 0x2764), (0x2795, 0x2797), 0x27A1, 0x27B0, 0x27BF, (0x2934, 0x2935),
    (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), 0x2B50, 0x2B55, 0x3030, 0x303D, 0x3297, 0x3299, 0x1F004,
    0x1F0CF, (0x1F170, 0x1F171), (0x1F17E, 0x1F17F), 0x1F18E, (0x1F191, 0x1F19A),
    (0x1F1E6, 0x1F1FF), (0x1F201, 0x1F202), 0x1F21A, 0x1F22F, (0x1F232, 0x1F23A),
    (0x1F250, 0x1F251), (0x1F300, 0x1F321), (0x1F324, 0x1F393), (0x1F396, 0x1F397),
    (0x1F399, 0x1F39B), (0x1F39E, 0x1F3F0), (0x1F3F3, 0x1F3F5), (0x1F3F7, 0x1F4FD),
    (0x1F4FF, 0x1F53D), (0x1F549, 0x1F54E), (0x1F550, 0x1F567), (0x1F56F, 0x1F570),
    (0x1F573, 0x1F57A), 0x1F587, (0x1F58A, 0x1F58D), 0x1F590, (0x1F595, 0x1F596),
    (0x1F5A4, 0x1F5A5), 0x1F5A8, (0x1F5B1, 0x1F5B2), 0x1F5BC, (0x1F5C2, 0x1F5C4),
    (0x1F5D1, 0x1F5D3), (0x1F5DC, 0x1F5DE), 0x1F5E1, 0x1F5E3, 0x1F5E8, 0x1F5EF, 0x1F5F3,
    (0x1F5FA, 0x1F64F), (0x1F680, 0x1F6C5), (0x1F6CB, 0x1F6D2), (0x1F6D5, 0x1F6D7),
    (0x1F6DC, 0x1F6E5), 0x1F6E9, (0x1F6EB, 0x1F6EC), 0x1F6F0, (0x1F6F3, 0x1F6FC),
    (0x1F7E0, 0x1F7EB), 0x1F7F0, (0x1F90C, 0x1F93A), (0x1F93C, 0x1F945), (0x1F947, 0x1F9FF),
    (0x1FA70, 0x1FA7C), (0x1FA80, 0x1FA88), (0x1FA90, 0x1FABD), (0x1FABF, 0x1FAC5),
    (0x1FACE, 0x1FADB), (0x1FAE0, 0x1FAE8), (0x1FAF0, 0x1FAF8),
) for c in (range(item[0], item[1] + 1) if isinstance(item, tuple) else (item,)))


def _plain(c: str) -> bool:
    """True for a character that is written as itself wherever it stands: not a control character
    other than a tab, CR and LF, not a line separator, a lone surrogate or a character of
    ``_UNSEEN``, and not a joiner or a selector, which are written as themselves in some places."""
    return (c in "\t\r\n" or " " <= c < "\x7f"
            or ("\x9f" < c and c not in "\u2028\u2029" and not "\ud800" <= c <= "\udfff"
                and c not in _UNSEEN and c not in _JOINERS and c not in _SELECTORS))


def _printable(text: str) -> str:
    """``text`` with every control character other than a tab, CR and LF written as an escape (ESC
    as ``\\x1b``), and with them the two Unicode line separators (``\\u2028``), a surrogate that
    stands alone (``\\udc80``), which cannot be written as UTF-8 and which some settings of stdout
    write as a raw byte, and the characters of ``_UNSEEN``: the ones that change the order text is
    shown in (``\\u202e``) and the ones that show nothing (``\\u200b``, a tag character as
    ``\\U000e0041``). A tab ends no line and is left. CR and LF are left for the place the text is
    printed, which deals with them in its own way.

    A variation selector is written as itself only straight after a character past U+009F that is
    itself written as itself and is no joiner or selector (an ideograph, a Mongolian letter); and
    U+FE0E and U+FE0F, which choose between a text and an emoji form, only after such a character
    that has the Emoji property in Unicode 15.0 (``_EMOJI``), or between a keycap's base and its
    U+20E3 (``1``, U+FE0F, U+20E3). A joiner is written as itself only between two characters past
    U+009F that are written as themselves, where the one before it may also be a selector that was
    written as itself and the one after it is no joiner or selector. Anywhere else they are written
    as escapes too (``\\ufe0f``, ``\\u200d``): a selector at the start of the text or after an
    ASCII character, an escape, a joiner or another selector; a joiner at an edge, beside an ASCII
    character or an escape, before a selector or beside another joiner."""
    out, before = [], ""            # before: what the character before was, where it was written as itself
    for i, c in enumerate(text):
        after = text[i + 1] if i + 1 < len(text) else ""
        if c in _STYLE_SELECTORS:
            kept = before == "emoji" or (before == "keycap base" and after == "\u20e3")
            before = "selector" if kept else ""
        elif c in _SELECTORS:
            kept = before in ("wide", "emoji")
            before = "selector" if kept else ""
        elif c in _JOINERS:
            kept = before in ("wide", "emoji", "selector") and after > "\x9f" and _plain(after)
            before = "joiner" if kept else ""
        else:
            kept = _plain(c)
            before = ("" if not kept else "emoji" if c in _EMOJI else "wide" if c > "\x9f"
                      else "keycap base" if c in _KEYCAP_BASES else "")
        out.append(c if kept else f"\\x{ord(c):02x}" if c < "\u0100" else f"\\u{ord(c):04x}" if c < "\U00010000"
                   else f"\\U{ord(c):08x}")
    return "".join(out)


def _failure_text(e) -> str:
    """What a failed fetch says of itself, fit to print. Control characters other than a tab, CR and
    LF, the two line separators, a lone surrogate and the characters of ``_UNSEEN`` are written as
    escapes, and a joiner or a selector that ``_printable``'s rule does not keep; then the token's
    value is replaced wherever the text holds it (an HTTP reason phrase can carry it back, and an
    escape can end in the character a token starts with, which is why the escapes are written
    before the token is looked for); then it is cut at 1,000 characters, so an escape that falls on
    the cut is left in part. ``_printable`` looks at the character after in two places, and the cut
    can take that character: a joiner the cut leaves as the last character has lost the one it
    joined to, and U+FE0E or U+FE0F left last after a keycap's base has lost its U+20E3. Either is
    written as an escape then, which makes the text 1,005 characters, and the token is looked for
    once more."""
    token = os.environ.get("GH_TOKEN")
    text = _hide(_printable(str(e)), token)[:1000]
    if text[-1:] in _JOINERS or (text[-1:] in _STYLE_SELECTORS and text[-2:-1] in _KEYCAP_BASES):
        text = _hide(text[:-1] + _printable(text[-1]), token)
    return text


def _command_data(text: str) -> str:
    """Text as a workflow command's message carries it: "%", CR and LF escaped, so text a server
    chose cannot end the command's line and start another."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _code_span(text: str) -> str:
    """Text as the step summary shows it between two backticks: on one line, with no backtick of its
    own to close the span."""
    return text.replace("`", "'").replace("\r", " ").replace("\n", " ")


def _message(text: str) -> str:
    """Text the pull request's author or a server chose, as the message of a workflow command: control
    characters as escapes, then "%", CR and LF escaped, so it stays the message of that one command."""
    return _command_data(_printable(text))


def _line(text: str) -> str:
    """Text as one line of the summary's prose carries it: control characters as escapes, CR and LF
    as spaces, and "|" and "<" written as the character references ``&#124;`` and ``&lt;``. Used for
    the reason of a gate that did not measure, which is words of the gate's own and a count."""
    return (_printable(text).replace("\r", " ").replace("\n", " ")
            .replace("|", "&#124;").replace("<", "&lt;"))


# cmark-gfm, the Markdown parser GitHub publishes, opens no code span with a string of more than 80
# backticks: text after such a string is read as Markdown and as tags, not as the span's text.
_SPAN_TICKS = 80


def _cell(text: str, table: bool = True) -> str:
    """Text the pull request's author chose, as one cell of the summary's table: a code span, so that
    it is shown as the characters it is and nothing in it is read as Markdown or as a tag. Control
    characters are written as escapes and CR and LF as spaces; a "|" is written ``\\|``, the table's
    own escape, which is read before the span is; the span opens and closes with one backtick more
    than the longest run of them in the text, and is padded with a space where the text begins or
    ends with a backtick, or begins and ends with a space (a reader takes one such pair of spaces
    off). A run of 80 or more backticks is written as that many ``\\x60``: the span's own string,
    one longer than the run, would be more than the 80 the parser opens a span with, and the cell
    would be read as Markdown and as tags. A text that is empty or all white space as ``str.strip``
    reads it (a tab and a no-break space included) gives an empty cell and no span. Rendered with
    cmark-gfm 0.29.0.gfm.2, a cell written so held its text and no element but the span (CHANGELOG,
    "a claim is shown in its cell as the characters it is"); GitHub's own page was not run. With
    ``table`` false the span stands in a paragraph, not a cell, and a "|" is written as it is: there
    the escape is not read and its backslash would be shown."""
    text = _printable(text).replace("\r", " ").replace("\n", " ")
    if table:
        text = text.replace("|", "\\|")
    if not text.strip():
        return ""
    text = re.sub("`{%d,}" % _SPAN_TICKS, lambda run: "\\x60" * len(run.group()), text)
    ticks = "`" * (max((len(run) for run in re.findall("`+", text)), default=0) + 1)
    pad = " " if text[0] == "`" or text[-1] == "`" or (text[0] == " " and text[-1] == " ") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


# GitHub documents 1 MiB as the most a step's summary may hold; past it the summary is not shown. The
# rows of the table are the one part whose size the pull request's author sets, so they stop here.
_ROWS_BYTES = 512 * 1024


# The most ``::warning`` commands the script prints for claims the gate could not check. A description
# gives a claim for each place one of the gate's templates matches, so its author sets how many. An
# ``::error`` command, which a contradicted claim has, and an uncheckable one under strict, is never
# left out. The most claims the gate did not verify in any description this repository records is 9.
_WARNINGS = 100


def _overlay_wrote(c) -> bool:
    """True for an UNCHECKABLE claim whose reason the PATH-2a overlay wrote: a kind it may move, and a
    reason that starts with its form."""
    return (c.verdict == "UNCHECKABLE" and c.kind in _OVERLAY_KINDS
            and isinstance(c.why, str) and _OVERLAY_WHY.match(c.why) is not None)


def _withheld_contradiction(c) -> bool:
    """True for a claim the gate reports UNCHECKABLE because the overlay withheld a CONTRADICTED
    reading of it. A blocking check fails on such a claim; one whose VERIFIED reading was withheld
    passes as any other UNCHECKABLE claim does."""
    return _overlay_wrote(c) and c.why.startswith("CONTRADICTED withheld")


def _by_weight(claims, withheld_first: bool = False) -> list:
    """``claims`` with the contradicted before the uncheckable and those before the rest, each kind
    in the order it had: the order of a table that cannot hold every claim. With ``withheld_first``
    (a blocking check), a claim whose CONTRADICTED reading the overlay withheld is ranked with the
    contradicted, since the job fails on it too."""
    def weight(c) -> int:
        if withheld_first and _withheld_contradiction(c):
            return 0
        return {"CONTRADICTED": 0, "UNCHECKABLE": 1}.get(c.verdict, 2)
    return sorted(claims, key=weight)


_NO_BASE = "0" * 40       # the `before` of a push that created its ref: there is no commit before it


def _creates_ref(event) -> bool:
    """True for a push that created its branch or tag: ``before`` is forty zeros, or ``created`` is
    true. Such a push has no base commit, so the compare address names none."""
    return event.get("before") == _NO_BASE or event.get("created") is True


def _claims_without_diff(summary: str, strict: bool):
    """How many claims the gate reads in ``summary`` with no diff to check them against, or None where
    asking raised. Which claims the gate reads turns on the sentence only, never on the diff, so this
    is the count it would read with the diff too. None is taken to mean the text holds a claim."""
    try:
        return len(gate_diff_text(summary, "", strict=strict).claims)
    except Exception:  # noqa: BLE001
        return None


def _claims_text(n) -> str:
    """The claims of a text the gate read with no diff, as a sentence names them."""
    return "the claims" if n is None else f"{n} claim(s)"


def _write_summary(lines) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        # newline="\n": the same bytes on every system; text mode on Windows would end each line in
        # CR LF, one byte more per row than the budget above counts.
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")


def _soft_fail(raw: str | None) -> tuple[bool, str | None]:
    """Read the soft-fail input. Returns ``(soft, unrecognised)``.

    Only an explicit "false" blocks. The value is compared case-insensitively with surrounding
    whitespace stripped: "false" gives ``(False, None)`` and "true" gives ``(True, None)``. Unset
    (``None``) is the input's default, "true". Every other value, the empty string included,
    reports as "true" does and comes back as ``unrecognised`` so the caller can name it in a
    warning.
    """
    if raw is None:
        return True, None
    value = raw.strip().lower()
    if value == "false":
        return False, None
    if value == "true":
        return True, None
    return True, raw


def main() -> int:
    strict = os.environ.get("STYXX_STRICT", "false").lower() == "true"
    # Only an explicit "false" blocks (see _soft_fail); any value it does not recognise reports, and is named here.
    soft, unrecognised = _soft_fail(os.environ.get("STYXX_SOFT_FAIL"))
    if unrecognised is not None:
        shown = repr(unrecognised).replace("%", "%25")      # repr escapes CR and LF; a workflow command needs % escaped
        print(f"::warning title=styxx diffgate - soft-fail value not recognised::soft-fail is {shown}, which is "
              "neither \"true\" nor \"false\" (compared case-insensitively, surrounding whitespace stripped), so "
              "this check reports and does not fail the job. Set soft-fail: \"false\" to fail it.")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = json.loads(open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8").read())

    if event_name == "pull_request":
        summary = (event.get("pull_request") or {}).get("body") or ""
        diff_url = (event.get("pull_request") or {}).get("url", "")
        number = (event.get("pull_request") or {}).get("number")
        # GitHub documents the number as an integer; anything else in the event file is shown as "?"
        what = f"PR #{number if type(number) is int else '?'} body"
        text_name = "description"
    elif event_name == "push":
        summary = (event.get("head_commit") or {}).get("message") or ""
        diff_url = (event.get("compare", "")
                    .replace("github.com", "api.github.com/repos", 1)
                    .replace("/compare/", "/compare/", 1))
        # canonical form: repository.compare_url template
        tpl = (event.get("repository") or {}).get("compare_url", "")
        if tpl:
            diff_url = tpl.replace("{base}", event.get("before", "")) \
                          .replace("{head}", event.get("after", ""))
        what = text_name = "head commit message"
        ref = event.get("ref")
        if isinstance(ref, str) and ref.startswith("refs/tags/") and _creates_ref(event):
            # A push that creates a tag has no base commit to compare with. Not gated in any mode, and
            # nothing is asked of the API.
            print("styxx diffgate: a push that creates a tag is not gated (it has no base commit to compare "
                  "with)")
            return 0
    else:
        print(f"styxx diffgate: event {event_name!r} not gated (pull_request/push only)")
        return 0

    if not summary.strip():
        print(f"styxx diffgate: empty {what} — nothing to gate")
        return 0
    try:
        diff = api(diff_url, "application/vnd.github.diff")
    except Exception as e:  # noqa: BLE001
        # This branch used to `return 0` under a comment saying "a broken fetch
        # must not fake a verdict" — and 0 IS the passing verdict. A gate that
        # could not read the diff has not cleared anything. A blocking check
        # (soft-fail "false") therefore fails here when the description holds a
        # claim the gate reads, and under `strict` whatever it holds; a
        # description with no claim the gate reads has nothing to judge, and
        # report mode passes, as it always did.
        # The text is urllib's, http.client's or this script's and can quote what a server sent (an
        # address, a reason phrase, a whole status line), so it is escaped for each place it goes: a
        # workflow command, and a code span. The token is looked for once more in each finished text,
        # because "%25" and "%0D" can end in the characters a token starts with, as an escape can.
        why, token = _failure_text(e), os.environ.get("GH_TOKEN")
        clean = ("_This is not a clean result. A gate with no diff to read "
                 "cannot contradict anything, which is true of every "
                 "summary ever written._")
        if soft:
            print(f"::warning title=styxx diffgate - could not fetch the diff::"
                  f"{_hide(_command_data(why), token)}. The gate did not run; this is not a pass.")
            _write_summary(["## styxx diffgate — DID NOT RUN", "",
                            f"The diff could not be fetched: `{_hide(_code_span(why), token)}`.", "",
                            clean, "",
                            f"_A workflow that sets `soft-fail: \"false\"` fails the job here when the {what} "
                            "holds a claim the gate reads, and with `strict: true` too whatever it holds._"])
            return 0
        n = _claims_without_diff(summary, strict)
        holds = n is None or n > 0
        no_base = (" A push that creates a branch has no base commit to compare with, so a re-run fails the "
                   "same way."
                   if event_name == "push" and _creates_ref(event) else "")
        again = ("A re-run reads the diff again, which is the remedy where the failure was passing; GitHub "
                 f"refuses a diff past its size limit, and no rewording of the {text_name} makes it serve one.")
        if not holds and not strict:
            print(f"::warning title=styxx diffgate - could not fetch the diff::"
                  f"{_hide(_command_data(why), token)}.{no_base} The gate did not run. The {what} holds no claim "
                  "the gate reads, so there was no claim to judge, and this blocking check passes.")
            _write_summary(["## styxx diffgate — DID NOT RUN", "",
                            f"The diff could not be fetched: `{_hide(_code_span(why), token)}`.{no_base}", "",
                            f"The {what} holds no claim the gate reads, so there was no claim to judge. This "
                            "workflow sets `soft-fail: \"false\"`; the job passes, as a blocking check does when "
                            f"the diff could not be read and the {text_name} holds no claim the gate reads.", "",
                            clean])
            print(f"styxx diffgate: DID NOT RUN - the diff could not be fetched; the {what} holds no claim "
                  "the gate reads, so the job passes")
            return 0
        if holds:
            rule = (f"The gate read no diff, so it judged none of {'the ' if n is not None else ''}"
                    f"{_claims_text(n)} it reads in the {what}. This check blocks (soft-fail: \"false\"), and a "
                    f"blocking check fails when the diff could not be read and the {what} holds a claim the gate "
                    "reads. Nothing was found contradicted.")
            remedy = (f"{again} A {what} that holds no claim the gate reads passes. The maintainer can set "
                      "soft-fail: \"true\" to report instead of block.")
        else:
            rule = ("The gate read no diff. This check blocks (soft-fail: \"false\") with strict on, and then a "
                    f"diff that could not be read fails the job whatever the {what} holds. Nothing was found "
                    "contradicted.")
            remedy = again
        print(f"::error title=styxx diffgate - diff not read so the check fails::"
              f"{_hide(_command_data(why), token)}.{no_base} {rule} {remedy}")
        _write_summary(["## styxx diffgate — DID NOT RUN, and the check fails", "",
                        f"The diff could not be fetched: `{_hide(_code_span(why), token)}`.{no_base}", "",
                        rule.replace('(soft-fail: "false")', '(`soft-fail: "false"`)') + " " + remedy, "",
                        clean])
        print("styxx diffgate: DID NOT RUN - the diff could not be fetched; "
              + (f"{_claims_text(n)} not judged; the job fails because soft-fail is \"false\"" if holds else
                 f"the {what} holds no claim the gate reads; the job fails because soft-fail is \"false\" and "
                 "strict is on"))
        return 1

    g = gate_diff_text(summary, diff, strict=strict)

    # `measured` is False when the fetched text yielded no file statuses and no
    # added lines — an empty diff, an error payload served with HTTP 200, an
    # HTML interstitial. PASS/FAIL cannot carry "this gate did not run".
    if not g.measured:
        # A blocking check fails here when the description holds a claim the gate reads (the gate
        # judged none of them), and under `strict` whatever it holds; report mode passes.
        n = len(g.claims)
        fails = not soft and (n > 0 or strict)
        head = ["## styxx diffgate — UNMEASURED", "",
                f"**The gate did not run.** {_line(g.why_unmeasured)}", "",
                f"Every claim in the {what} is reported UNCHECKABLE, not "
                "verified. A PASS here would mean *nothing contradicted "
                "the summary* — which is true of any summary when there "
                "is no diff to read.", ""]
        if soft:
            print(f"::warning title=styxx diffgate - UNMEASURED::"
                  f"{_message(g.why_unmeasured)}. The gate did not run; this is not a pass.")
            _write_summary(head + [f"_A workflow that sets `soft-fail: \"false\"` fails the job on this when the "
                                   f"{what} holds a claim the gate reads, and with `strict: true` too whatever "
                                   "it holds._"])
            return 0
        if not fails:
            print(f"::warning title=styxx diffgate - UNMEASURED::"
                  f"{_message(g.why_unmeasured)}. The gate did not run. The {what} holds no claim the gate "
                  "reads, so there was no claim to judge, and this blocking check passes.")
            _write_summary(head + [f"_The {what} holds no claim the gate reads, so there was no claim to judge. "
                                   "This workflow sets `soft-fail: \"false\"`; the job passes, as a blocking check "
                                   f"does when the gate did not measure the diff and the {text_name} holds no "
                                   "claim the gate reads._"])
            print(f"styxx diffgate: UNMEASURED - gate verdict {g.verdict}; the {what} holds no claim the gate "
                  "reads, so the job passes")
            return 0
        if n:
            rule = (f"The gate read no file and no added line, so it judged none of the {n} claim(s) in the "
                    f"{what}. This check blocks (soft-fail: \"false\"), and a blocking check fails when the gate "
                    f"did not measure the diff and the {what} holds a claim the gate reads. The gate's own "
                    f"verdict is {g.verdict}; nothing was found contradicted.")
            remedy = ("A re-run reads the diff again, which is the remedy where the failure was passing. A "
                      f"{what} that holds no claim the gate reads passes. The maintainer can set soft-fail: "
                      "\"true\" to report instead of block.")
        else:
            rule = ("The gate read no file and no added line. This check blocks (soft-fail: \"false\") with "
                    f"strict on, and then a diff the gate did not measure fails the job whatever the {what} "
                    f"holds. The gate's own verdict is {g.verdict}; nothing was found contradicted.")
            remedy = "A re-run reads the diff again, which is the remedy where the failure was passing."
        print(f"::error title=styxx diffgate - UNMEASURED so the check fails::"
              f"{_message(g.why_unmeasured)}. {rule} {remedy}")
        _write_summary(head + ["_" + rule.replace('(soft-fail: "false")', '(`soft-fail: "false"`)') + " "
                               + remedy + "_"])
        print(f"styxx diffgate: UNMEASURED - gate verdict {g.verdict}; "
              + (f"{n} claim(s) not judged; the job fails because soft-fail is \"false\"" if n else
                 f"the {what} holds no claim the gate reads; the job fails because soft-fail is \"false\" and "
                 "strict is on"))
        return 1

    # A blocking check fails on a claim whose CONTRADICTED reading the PATH-2a overlay withheld: the gate
    # did not judge it, and a blocking check does not pass that one. The claim stays UNCHECKABLE and is never
    # called contradicted here; the gate's verdict word is the gate's and is not changed.
    withheld = [c for c in g.claims if _withheld_contradiction(c)] if not soft else []
    if withheld and g.verdict == "PASS":
        verdict_line = (f"**{g.verdict}** is the gate's verdict, and the job fails · {len(g.claims)} claim(s) "
                        f"checked on the {what}")
    else:
        verdict_line = f"**{g.verdict}** · {len(g.claims)} claim(s) checked on the {what}"
    lines = ["## styxx diffgate — the summary vs the diff", "", verdict_line, ""]
    if withheld:
        # Written above the table and outside its byte budget, so the reason the job fails is never cut.
        # At most ten of the claims are named, each cut at 80 characters, so these lines stay small.
        shown = ", ".join(_cell(c.text[:80], table=False) or "(empty)" for c in withheld[:10])
        more = f", and {len(withheld) - 10} more" if len(withheld) > 10 else ""
        although = ", although the gate's verdict is PASS" if g.verdict == "PASS" else ""
        lines += [f"**The job fails on {len(withheld)} claim(s) the gate did not judge{although}:** the gate's "
                  "overlay (PATH-2a) withheld "
                  "a CONTRADICTED reading of each, because the gate's reading in that place may be wrong. They "
                  "are marked ❓ UNCHECKABLE below, and they are not found contradicted. This workflow sets "
                  "`soft-fail: \"false\"`, and a blocking check fails on a claim withheld this way. To pass, "
                  "remove the claim or reword it; where the overlay's reason names a character in the summary, "
                  "writing the claim in plain ASCII lets both of the gate's readers read it alike. The "
                  f"maintainer can set `soft-fail: \"true\"`. The claim(s): {shown}{more}.", ""]
    if g.claims:
        lines += ["| verdict | claim | evidence |", "|---|---|---|"]
        def row_of(c) -> str:
            mark = {"VERIFIED": "✅", "CONTRADICTED": "❌", "UNCHECKABLE": "❓"}[c.verdict]
            # PATH-2a (NOTE_path2a_second_pass_2026_09_30): a reason the overlay wrote is shown whole, since its
            # leading 100 characters are the withheld verdict and the phrase, and main's reading comes after them.
            # Only a reason the overlay wrote (NOTE_path2a_third_pass_2026_09_30, A-2): an UNCHECKABLE claim of a
            # kind it may move, whose reason starts with its form. No reason main writes for those kinds starts so;
            # a reason that only contains the words, such as a DECLARE-1 MALFORMED one, is cut as main cuts it.
            # I-3 (NOTE_path2a_sixth_pass_2026_09_30): the overlay's own words whole, and main's reading after them cut
            # as main cuts a reason, at 100 characters, so a long path cannot carry the table past GitHub's step
            # summary limit.
            if _overlay_wrote(c):
                head, sep, mains = c.why.partition(_OVERLAY_MAIN)
                why = head + sep + mains[:100]
            else:
                why = c.why[:100]
            # A claim's text is the pull request author's, and a reason can quote it and the diff's paths, so
            # each is written as a code span: a "|" or a line break of theirs cannot end the cell or start a
            # row, and nothing of theirs is read as Markdown or as a tag.
            return f"| {mark} {c.verdict} | {_cell(c.text[:80])} | {_cell(why)} |"

        def rows_that_fit(claims) -> list:
            """The rows of ``claims``, in that order, up to the one that would take the rows past their
            budget. No row is written for a claim after it."""
            rows, used = [], 0
            for c in claims:
                row = row_of(c)
                used += len(row.encode("utf-8")) + 1
                if used > _ROWS_BYTES:
                    break
                rows.append(row)
            return rows

        # The rows are in the order the claims were read where they all fit. Where they do not, the
        # contradicted claims are taken before the uncheckable and those before the verified, each kind in
        # the order read, so that claims of no weight cannot fill the table ahead of a contradicted one.
        # The rows' bytes are the same in any order, so a table that stopped once stops again.
        rows, left = rows_that_fit(g.claims), []
        if len(rows) < len(g.claims):
            ordered = _by_weight(g.claims, withheld_first=bool(withheld))
            rows = rows_that_fit(ordered)
            left = ordered[len(rows):]
        lines += rows
        if left:
            count = {v: sum(1 for c in left if c.verdict == v) for v in ("CONTRADICTED", "UNCHECKABLE", "VERIFIED")}
            lines += ["", f"_{len(left)} more claim(s) are not in this table, which stops at {_ROWS_BYTES:,} bytes "
                          "of rows to stay under the size GitHub allows a step's summary: "
                          f"{count['CONTRADICTED']} CONTRADICTED, {count['UNCHECKABLE']} UNCHECKABLE, "
                          f"{count['VERIFIED']} VERIFIED. Its rows are written with the CONTRADICTED before the "
                          "UNCHECKABLE, and those before the VERIFIED, each kind in the order read, which may not "
                          "be the order the claims were read. "
                          + ("In this blocking check an UNCHECKABLE claim whose CONTRADICTED reading the overlay "
                             "withheld is written with the CONTRADICTED. " if withheld else "")
                          + "The verdict above counts every claim._"]
    else:
        lines += ["_No diff-shaped claims found. The gate checks a closed template set "
                  "(touched/created/deleted paths, added functions/tests, file counts, "
                  "only-touches scopes, tests-pass); prose outside it is not judged._"]
    # This footer used to say "zero false accusations across both public validation
    # corpora". The README withdrew that claim (re-run at 7.46.0 the committed sweep
    # found four, all false accusations), and a footer that kept asserting it on every
    # gated PR in every adopter's repo was the same present-tense mistake one level down.
    # Which of the two the job did is said here, because the default changed: soft-fail reports
    # unless the workflow sets it to "false" (action.yml; CHANGELOG "the GitHub Action reports by
    # default"). The figures are the receipts' own; the CHANGELOG entry says where each comes from.
    rule = ("on a ❌; on a ❓ whose CONTRADICTED reading the gate's overlay withheld; and on a diff it could "
            f"not read or did not measure while the {text_name} holds a claim the gate reads. On every other ❓ "
            "it fails only with `strict: true`. ")
    if not soft:
        mode = "_This workflow sets `soft-fail: \"false\"`, so the job fails " + rule
    elif unrecognised is not None:
        mode = ("_soft-fail has a value this Action does not recognise (the warning annotation names "
                "it), so this check reports and does not fail the job, as it does by default; only "
                "`soft-fail: \"false\"` fails it, " + rule)
    else:
        mode = ("_soft-fail is on, as it is by default, so this check reports and does not fail the "
                "job; a workflow that sets `soft-fail: \"false\"` fails it " + rule)
    p = "https://github.com/fathom-lab/styxx/blob/main/papers/closed-model-frontier/"
    lines += ["", mode + "A path claim the diff does not show is reported ❓ UNCHECKABLE, not "
              "accused: the 100 accusations EXTERNAL-1 sampled from 71,016 external agent-authored "
              "PRs, 85 of them path claims, measured precision 0.23 against a preregistered 0.95 "
              f"floor ([EXTERNAL-1]({p}RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md), "
              f"2026-08-31; receipts [`external1_adjudication.json`]({p}external1_adjudication.json), "
              f"[`external1_summary.json`]({p}external1_summary.json)), and a repaired path accuser "
              "measured 0.16 on a held-out sample "
              f"([V14]({p}RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md), 2026-09-01; "
              f"receipt [`v14_adjudication.json`]({p}v14_adjudication.json)), so that accusation is "
              "withheld until a held-out repair clears the floor. Of the kinds that still accuse, "
              "`only_touches` is the one with a measured precision: 0.25, 2 of 8 accusations "
              f"correct ([PATH-1]({p}RESULT_path1_only_touches_repair_2026_09_17.md), 2026-09-17, "
              f"re-derived by [SCOPE-1]({p}RESULT_scope1_ABANDONED_2026_09_18.md), 2026-09-18; "
              f"receipt [`scope1_footprint.json`]({p}scope1_footprint.json)). The withdrawn "
              "zero-false-accusation claim, and what replaced it, are in the "
              "[styxx README](https://github.com/fathom-lab/styxx/blob/main/README.md)._"]
    _write_summary(lines)

    # Each claim the gate did not verify has a command, in the order the claims were read, but that the
    # warnings stop at _WARNINGS: an uncheckable claim past that many has none, and one command says how
    # many there are. An error is never left out, so a contradicted claim always has its command, and
    # under strict an uncheckable one has too.
    failing = False
    warned = unwarned = 0
    for c in g.claims:
        if c.verdict == "CONTRADICTED":
            failing = True
            print(f"::error title=styxx diffgate - contradicted claim::{_message(c.text[:120])} - {_message(c.why)}")
        elif not soft and _withheld_contradiction(c):
            # Blocking only: the claim is not called contradicted; the overlay's own reason is quoted as it is.
            failing = True
            print(f"::error title=styxx diffgate - withheld so the check fails::{_message(c.text[:120])} - The "
                  "gate did not judge this claim: its overlay (PATH-2a) withheld a CONTRADICTED reading of it, "
                  "because the gate's reading in this place may be wrong. This check blocks (soft-fail: "
                  "\"false\"), and a blocking check fails on a claim withheld this way; every other UNCHECKABLE "
                  "claim passes unless strict is on. To pass, remove the claim or reword it; where the overlay's "
                  "reason names a character in the summary, writing the claim in plain ASCII lets both of the "
                  "gate's readers read it alike. The maintainer can set soft-fail: \"true\". The overlay's "
                  f"reason: {_message(c.why)}")
        elif c.verdict == "UNCHECKABLE" and strict:
            failing = True
            print(f"::error title=styxx diffgate - uncheckable under strict::{_message(c.text[:120])} - "
                  f"{_message(c.why)}")
        elif c.verdict == "UNCHECKABLE" and warned < _WARNINGS:
            warned += 1
            print(f"::warning title=styxx diffgate - uncheckable::{_message(c.text[:120])} - {_message(c.why)}")
        elif c.verdict == "UNCHECKABLE":
            unwarned += 1
    if unwarned:
        print(f"::warning title=styxx diffgate - more uncheckable claims than warnings::{unwarned} more claim(s) "
              f"that the gate could not check have no warning here, where {_WARNINGS} is the most that are "
              "printed. A contradicted claim always has its command. The verdict below counts every claim.")
    last = (f"styxx diffgate: {g.verdict} - {len(g.claims)} claim(s), "
            f"{sum(1 for c in g.claims if c.verdict == 'CONTRADICTED')} contradicted")
    if withheld:
        last += (f"; {len(withheld)} not judged (a CONTRADICTED reading withheld by the overlay), so the job "
                 + ("fails although the gate's verdict is PASS" if g.verdict == "PASS" else "fails")
                 + " because soft-fail is \"false\"")
    print(last)
    if failing and not soft:
        return 1
    if failing and soft:
        print("::warning::styxx diffgate found failures but soft-fail is on, as it is by default, so "
              "the job passes; set soft-fail: \"false\" to fail it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
