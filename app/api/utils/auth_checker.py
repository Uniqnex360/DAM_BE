def check_authorized(record_user_id, current_user) -> bool:
    return str(record_user_id) == str(current_user.id) or getattr(current_user, "role", None) == "admin"