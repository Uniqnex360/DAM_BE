from app.models.auth import User
import logging

logger = logging.getLogger('get_target_user_id')


def get_target_user_id(
    current_user: User,
    requested_user_id: str = None
) -> str:
    
    is_admin = current_user.role == "admin"

    if requested_user_id and is_admin:
        logger.info(
            f"Admin {current_user.email} accessing data for user {requested_user_id}")
        return requested_user_id

    return current_user.id
