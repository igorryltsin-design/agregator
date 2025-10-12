import pytest

from app import app, facet_service
from agregator.services.facets import FacetQueryParams
from models import Collection, File, Tag, db


def prepare_app_context():
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['TESTING'] = True
    with app.app_context():
        db.engine.dispose()
        db.drop_all()
        db.create_all()
        col = Collection(name='Test', slug='test', searchable=True, graphable=True)
        db.session.add(col)
        db.session.commit()
        return col.id


def test_facets_empty_scope():
    col_id = prepare_app_context()
    params = FacetQueryParams(
        query='',
        material_type='',
        context='search',
        include_types=True,
        tag_filters=[],
        collection_filter=col_id,
        allowed_scope={col_id},
        allowed_keys_list=None,
        year_from='',
        year_to='',
        size_min='',
        size_max='',
        sources={'tags': True},
        request_args=(),
    )
    with app.app_context():
        facet_service.invalidate('test')
        result = facet_service.get_facets(
            params,
            search_candidate_fn=lambda q, limit=0: [],
            like_filter_fn=lambda query, text: text,
        )
    assert result['context'] == 'search'
    assert 'tag_facets' in result
    assert isinstance(result['tag_facets'], dict)
