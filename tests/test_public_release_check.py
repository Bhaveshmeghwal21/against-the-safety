"""Public release checks must not fall back to a cached login or preview items."""
from types import SimpleNamespace
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import publish_release as release


def test_public_check_reads_complete_collection_anonymously():
    spec = {'title': 'test collection', 'items': [('model', 0)]}
    repo = [('model', 'model-name', 'model')]
    preview = SimpleNamespace(title=spec['title'], slug='owner/collection', items=[])
    full = SimpleNamespace(slug=preview.slug, private=False,
                           items=[SimpleNamespace(item_type='model', item_id='owner/model-name')])
    api = SimpleNamespace(model_info=lambda _: SimpleNamespace(gated=False, private=False),
                          dataset_info=lambda _: None, space_info=lambda _: None,
                          list_collections=lambda **_: [preview], get_collection=lambda _: full)
    with patch.object(release, 'REPOS', repo), patch.object(release, 'COLLECTIONS', {'models': spec}), \
         patch.object(release, 'HfApi', return_value=api) as factory, \
         patch.object(release.requests, 'get', return_value=SimpleNamespace(status_code=200)):
        assert release.check('owner')
        factory.assert_called_once_with(token=False)
        full.items.clear()
        assert not release.check('owner')
