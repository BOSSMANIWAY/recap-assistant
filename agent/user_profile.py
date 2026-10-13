"""
User Profile - персональные данные для автозаполнения форм

Используется в промпте агента: когда задача требует ввести личные данные
(регистрация, заказ, контактная форма), LLM берёт их отсюда, а не выдумывает.

Заполните значения под себя — они передаются в промпт LLM при заполнении форм.
Пустые поля агент игнорирует и запросит данные у пользователя.
ВАЖНО: не коммитьте заполненный файл в публичный репозиторий.
"""

USER_PROFILE = {
    # Контакты
    "phone": "",
    "phone_formatted": "",
    "email": "",
    "email_backup": "",

    # Имя
    "full_name": "",
    "first_name": "",
    "last_name": "",
    "middle_name": "",
    "first_name_en": "",
    "last_name_en": "",

    # Никнеймы / логины
    "username": "",
    "nickname": "",

    # Сайты пользователя
    "website": "",
}


def profile_to_prompt() -> str:
    """Текстовое представление профиля для промпта LLM"""
    return f"""=== ДАННЫЕ ПОЛЬЗОВАТЕЛЯ (используй при заполнении форм) ===
Телефон: {USER_PROFILE['phone']} (или {USER_PROFILE['phone_formatted']} — выбери формат, который требует поле)
Email основной: {USER_PROFILE['email']}
Email резервный: {USER_PROFILE['email_backup']}
ФИО: {USER_PROFILE['full_name']}
Имя: {USER_PROFILE['first_name']} / {USER_PROFILE['first_name_en']}
Фамилия: {USER_PROFILE['last_name']} / {USER_PROFILE['last_name_en']}
Отчество: {USER_PROFILE['middle_name']}
Никнейм/логин: {USER_PROFILE['username']}
Сайт: {USER_PROFILE['website']}

Правила:
- Выбирай значение по смыслу поля (name→имя, phone→телефон, email→email, username→ник)
- Если поле требует латиницу — используй английские варианты имени
- Если данных не хватает (пароль, карта, адрес, код из SMS) — НЕ выдумывай,
  запроси у пользователя через finish с reason "нужен X"
- Пароли не хранятся и не вводятся автоматически без явного указания в задаче"""
