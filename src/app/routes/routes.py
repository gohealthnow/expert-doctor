# Define as rotas/endpoints da API

from flask import Blueprint, request, jsonify
from ..ia import process_text

bp = Blueprint('main', __name__)

@bp.route('/api/checklist', methods=['POST'])
def get_checklist():
    data = request.get_json()
    raw_text = data.get('text')
    checklist = process_text(raw_text)
    return jsonify({'checklist': checklist})

@bp.route('/api/form', methods=['POST'])
def get_form():
    data = request.get_json()
    raw_text = data.get('text')
    form = generate_form(raw_text)
    return jsonify({'form': form})

def generate_form(text):
    # Implement your logic to generate a form based on the input text
    # This is a placeholder implementation
    return {
        'questions': [
            {'question': 'How do you feel?', 'type': 'text'},
            {'question': 'Rate your pain level from 1 to 10', 'type': 'number'},
            {'question': 'Describe your symptoms', 'type': 'textarea'}
        ]
    }
