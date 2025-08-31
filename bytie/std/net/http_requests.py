import uuid
import requests
from requests import Session
from typing import Optional, Dict, Any

from bytie.errors import BytieError
from bytie.types import ErrorVal, MapVal, TypeSpec


class HttpRequests:
    def __init__(self):
        self.sessions = {}

    def format_response(self, resp: requests.Response) -> MapVal:
        return MapVal(value_type=TypeSpec.any(), 
                      entries={
                        "status": resp.status_code,
                        "headers": dict(resp.headers),
                        "body": resp.text
                     })

    def create_session(self, session_kwargs: Dict[str, Any]) -> str:    
        print(f"Creating session with kwargs: {session_kwargs}")
        session = requests.Session(**session_kwargs)
        session_uuid = str(uuid.uuid4())
        self.sessions[session_uuid] = session
        return session_uuid

    def get_session(self, session_id: str) -> str:
        if session_id not in self.sessions:
            raise BytieError(ErrorVal(name="NetworkSessionError", 
                                      message=f"Session {session_id} not found"))
        return self.sessions[session_id]

    def get(self, url: str, session_id: str) -> MapVal:
        if session_id == '':
            resp = requests.get(url)
        else:
            session = self.get_session(session_id)
            resp = session.get(url)
        return self.format_response(resp)

    def post(self, url: str, data: dict, session_id: str) -> MapVal:
        if session_id == '':
            resp = requests.post(url, json=data)
        else:
            session = self.get_session(session_id)
            resp = session.post(url, json=data)
        return self.format_response(resp)

    def put(self, url: str, data: dict, session_id: str) -> MapVal:
        if session_id == '':
            resp = requests.put(url, data)
        else:
            session = self.get_session(session_id)
            resp = session.put(url, data)
        return self.format_response(resp)

    def delete(self, url: str, session_id: str) -> MapVal:
        if session_id == '':
            resp = requests.delete(url)
        else:
            session = self.get_session(session_id)
            resp = session.delete(url)
        return self.format_response(resp)