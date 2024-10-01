# Define as rotas/endpoints da API

from flask import Blueprint, request, jsonify
from .ia import process_text

bp = Blueprint('main', __name__)

@bp.route('/api/checklist', methods=['POST'])
def get_checklist():
    data = request.get_json()
    raw_text = data.get('text')
    checklist = process_text(raw_text)
    return jsonify({'checklist': checklist})
