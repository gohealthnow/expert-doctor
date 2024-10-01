# Define as rotas/endpoints da API

from flask import Blueprint, request, jsonify
from .ia import processa_texto

bp = Blueprint('main', __name__)

@bp.route('/api/checklist', methods=['POST'])
def get_checklist():
    data = request.get_json()
    texto = data.get('texto', '')
    checklist = processa_texto(texto)
    return jsonify({'checklist': checklist})
