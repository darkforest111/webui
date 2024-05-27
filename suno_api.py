import requests
import time
from http.cookies import SimpleCookie
from threading import Thread

header={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Origin":"https://suno.com",
    "Referer":"https://suno.com",
    'Content-Type': 'application/json',
}
COOKIE=""
SESSION_ID=""
token=""


class SunoCookie:
    def __init__(self):
        self.cookie = SimpleCookie()
        self.session_id = None
        self.token = None

    def load_cookie(self, cookie_str):
        self.cookie.load(cookie_str)

    def get_cookie(self):
        return ";".join([f"{i}={self.cookie.get(i).value}" for i in self.cookie.keys()])

    def set_session_id(self, session_id):
        self.session_id = session_id

    def get_session_id(self):
        return self.session_id

    def get_token(self):
        return self.token

    def set_token(self, token: str):
        self.token = token


def update_token(suno_cookie: SunoCookie):
    global header,token
    header.update({"Cookie": suno_cookie.get_cookie()})
    session_id = suno_cookie.get_session_id()
    resp = requests.post(
        url=f"https://clerk.suno.com/v1/client/sessions/{session_id}/tokens?_clerk_js_version=4.72.0-snapshot.vc141245",
        headers=header,
    )
    resp_headers = dict(resp.headers)
    set_cookie = resp_headers.get("Set-Cookie")
    suno_cookie.load_cookie(set_cookie)
    token = resp.json().get("jwt")
    suno_cookie.set_token(token)


def keep_alive(suno_cookie: SunoCookie):
    global token
    try:
        update_token(suno_cookie)
    except Exception as e:
         print(e)

# suno_auth = SunoCookie()
# suno_auth.set_session_id(SESSION_ID)
# suno_auth.load_cookie(COOKIE)
# keep_alive(suno_auth)
# print(token)

    # while True:
    #     try:
    #         update_token(suno_cookie)
    #     except Exception as e:
    #         print(e)
    #     finally:
    #         time.sleep(5)


# def start_keep_alive(suno_cookie: SunoCookie):
#     t = Thread(target=keep_alive, args=(suno_cookie,))
#     t.start()

# start_keep_alive(suno_auth)

