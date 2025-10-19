import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app, _compute_authority_snapshot, AUTHORITY_GRAPH_CACHE, _rebuild_feedback_model, _get_feedback_weights
from models import db, Collection, File, Tag, AiSearchKeywordFeedback, AiSearchFeedbackModel


def prepare_context():
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['TESTING'] = True
    if 'sqlalchemy' not in app.extensions:
        db.init_app(app)
    with app.app_context():
        db.engine.dispose()
        db.drop_all()
        db.create_all()
        col1 = Collection(name='Alpha', slug='alpha', searchable=True, graphable=True)
        col2 = Collection(name='Beta', slug='beta', searchable=True, graphable=True)
        db.session.add_all([col1, col2])
        db.session.commit()
        file1 = File(
            collection_id=col1.id,
            path='doc1.pdf',
            rel_path='doc1.pdf',
            filename='doc1.pdf',
            title='Документ 1',
            author='Alice Smith',
            keywords='ml, ai',
            year='2022',
        )
        file2 = File(
            collection_id=col1.id,
            path='doc2.pdf',
            rel_path='doc2.pdf',
            filename='doc2.pdf',
            title='Документ 2',
            author='Alice Smith',
            keywords='ml, robotics',
            year='2021',
        )
        file3 = File(
            collection_id=col2.id,
            path='doc3.pdf',
            rel_path='doc3.pdf',
            filename='doc3.pdf',
            title='Документ 3',
            author='Bob Lee',
            keywords='physics',
            year='2020',
        )
        db.session.add_all([file1, file2, file3])
        db.session.commit()
        db.session.add_all([
            Tag(file_id=file1.id, key='topic', value='AI'),
            Tag(file_id=file2.id, key='topic', value='AI'),
            Tag(file_id=file2.id, key='topic', value='Robotics'),
            Tag(file_id=file3.id, key='topic', value='Science'),
        ])
        db.session.commit()
        return file1.id, file2.id, file3.id


def test_authority_snapshot_produces_scores():
    fid1, fid2, fid3 = prepare_context()
    with app.app_context():
        AUTHORITY_GRAPH_CACHE.clear()
        snapshot = _compute_authority_snapshot(None)
    assert snapshot and snapshot.get('doc_scores'), 'authority snapshot should produce document scores'
    doc_scores = snapshot['doc_scores']
    assert fid1 in doc_scores and fid2 in doc_scores
    assert doc_scores[fid1] == doc_scores[fid2]
    assert doc_scores[fid1] > doc_scores.get(fid3, 0.0)

    authors = {entry['name']: entry for entry in snapshot.get('author_entries', []) if isinstance(entry, dict)}
    assert 'Alice Smith' in authors
    assert authors['Alice Smith']['score'] > authors.get('Bob Lee', {}).get('score', 0.0)

    topics = {f"{entry.get('key')}:{entry.get('label')}": entry for entry in snapshot.get('topic_entries', []) if isinstance(entry, dict)}
    assert 'topic:AI' in topics
    assert topics['topic:AI']['score'] > topics.get('topic:Science', {}).get('score', 0.0)


def test_feedback_training_produces_weights():
    fid1, fid2, fid3 = prepare_context()
    with app.app_context():
        db.session.add_all([
            AiSearchKeywordFeedback(file_id=fid1, query_hash='hash-a', action='relevant'),
            AiSearchKeywordFeedback(file_id=fid1, query_hash='hash-a', action='click'),
            AiSearchKeywordFeedback(file_id=fid2, query_hash='hash-a', action='irrelevant'),
            AiSearchKeywordFeedback(file_id=fid3, query_hash='hash-b', action='relevant'),
        ])
        db.session.commit()
        stats = _rebuild_feedback_model()
        assert stats['files'] >= 2
        weights = _get_feedback_weights()
        assert fid1 in weights
        assert fid2 in weights
        assert weights[fid1]['positive'] >= 1
        assert weights[fid1]['weight'] > 0
        assert weights[fid2]['negative'] >= 1
        assert weights[fid2]['weight'] < 0
        row = AiSearchFeedbackModel.query.filter_by(file_id=fid1).first()
        assert row is not None
        assert row.positive >= 1
        assert row.clicks >= 1
        assert row.weight > weights[fid2]['weight']
