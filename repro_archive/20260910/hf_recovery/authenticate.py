"""Read a credential without echo and replace the standard HF cache after validation."""
import getpass
import json
from pathlib import Path

from huggingface_hub import HfApi, constants
from huggingface_hub.utils._auth import _save_token


def main():
    token = getpass.getpass('Hugging Face credential (hidden): ').strip()
    # Chat formatting can escape the underscore in an otherwise valid token.
    token = token.replace('\\_', '_')
    assert token.startswith('hf_') and not any(c.isspace() for c in token)
    try:
        account = HfApi(token=token).whoami()
    except Exception as error:
        # Do not serialize a request, response headers, or the supplied secret.
        raise SystemExit('HF authentication failed: ' + type(error).__name__) from None
    assert account['name'] == 'razaxq', 'Unexpected HF account; credential not saved'
    _save_token(token, token_name='BetterLViT-upload-20260910')
    target = Path(constants.HF_TOKEN_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(token, encoding='utf-8')
    print(json.dumps(dict(authenticated=True, account=account['name'], credential_saved_to_standard_hf_cache=True)))


if __name__ == '__main__':
    main()
