from server.services.notification_service import get_notification_service
from tests.utils.sse import next_payload


def test_notification_service_broadcast_and_receive():
    ns = get_notification_service()
    account_id = 42
    lang = "en"

    # Subscribe and send an event
    q = ns.subscribe(account_id, lang)
    ns.send_translations_ready(account_id, lang, text_id=777)

    # Next non-handshake event should be our translations_ready
    ev = next_payload(q)
    assert ev is not None
    assert ev.type == "translations_ready"
    assert isinstance(ev.data, dict)
    assert ev.data.get("text_id") == 777
