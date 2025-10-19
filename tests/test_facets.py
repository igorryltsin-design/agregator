import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app, facet_service
from agregator.services.facets import FacetQueryParams
from models import Collection, File, Tag, db


def prepare_app_context():
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['TESTING'] = True
    if 'sqlalchemy' not in app.extensions:
        db.init_app(app)
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
        sources={'tags': True, 'authors': True, 'years': True},
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
    assert 'authors' in result
    assert 'years' in result
    assert 'suggestions' in result


def test_facets_metadata_blocks():
    col_id = prepare_app_context()
    with app.app_context():
        file1 = File(
            collection_id=col_id,
            path='/tmp/doc1.pdf',
            rel_path='doc1.pdf',
            filename='doc1.pdf',
            author='Alice Smith',
            year='2021',
            material_type='article',
        )
        file2 = File(
            collection_id=col_id,
            path='/tmp/doc2.pdf',
            rel_path='doc2.pdf',
            filename='doc2.pdf',
            author='Bob Lee',
            year='2020',
            material_type='preprint',
        )
        db.session.add_all([file1, file2])
        db.session.commit()
        tag = Tag(file_id=file1.id, key='topic', value='LLM')
        db.session.add(tag)
        db.session.commit()

        params = FacetQueryParams(
            query='',
            material_type='',
            context='search',
            include_types=True,
            tag_filters=['topic=LLM'],
            collection_filter=col_id,
            allowed_scope={col_id},
            allowed_keys_list=None,
            year_from='',
            year_to='',
            size_min='',
            size_max='',
            sources={'tags': True, 'authors': True, 'years': True},
            request_args=(),
        )
        facet_service.invalidate('metadata-test')
        result = facet_service.get_facets(
            params,
            search_candidate_fn=lambda q, limit=0: None,
            like_filter_fn=lambda query, text: query,
        )

    authors = {tuple(entry) for entry in result.get('authors', [])}
    years = {tuple(entry) for entry in result.get('years', [])}
    suggestions = result.get('suggestions', [])

    assert ('Alice Smith', 1) in authors
    assert ('2021', 1) in years
    assert any(sug.get('kind') == 'author' and sug.get('value') == 'Alice Smith' for sug in suggestions)
    assert any(sug.get('kind') == 'year' and sug.get('value') == '2021' for sug in suggestions)
