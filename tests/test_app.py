import pytest
from app import app  # Importez votre application Flask

@pytest.fixture
def client():
    # Configure Flask pour le test
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    # Teste si la page d'accueil se charge correctement
    response = client.get('/')
    assert response.status_code == 200
    assert b"Bienvenue" in response.data  # Vérifie que le texte "Bienvenue" est dans la page

def test_404_page(client):
    # Teste une page qui n'existe pas
    response = client.get('/page-qui-n-existe-pas')
    assert response.status_code == 404
