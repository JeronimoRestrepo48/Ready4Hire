#!/usr/bin/env python3
"""
Script de verificación de profesiones y preguntas de contexto.
Verifica que todas las profesiones tengan 5 preguntas de contexto.
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain.value_objects.context_questions import (
    PROFESSION_NAME_MAPPING,
    PROFESSION_CONTEXT_QUESTIONS,
    CONTEXT_QUESTIONS_COUNT,
)
from app.domain.entities.profession import PROFESSIONS_DATABASE, get_all_professions


def verify_context_questions():
    """Verifica que todas las profesiones tengan preguntas de contexto."""
    print("=" * 80)
    print("VERIFICACIÓN DE PREGUNTAS DE CONTEXTO POR PROFESIÓN")
    print("=" * 80)
    
    issues = []
    missing_professions = []
    incomplete_professions = []
    
    # Obtener todas las profesiones del mapeo
    all_profession_names = set(PROFESSION_NAME_MAPPING.keys())
    all_profession_keys = set(PROFESSION_NAME_MAPPING.values())
    
    print(f"\n📊 Total profesiones en mapeo: {len(all_profession_names)}")
    print(f"📊 Total profesiones en base de datos: {len(PROFESSIONS_DATABASE)}")
    print(f"📊 Total profesiones con preguntas de contexto: {len(PROFESSION_CONTEXT_QUESTIONS)}")
    print(f"📊 Preguntas de contexto esperadas por profesión: {CONTEXT_QUESTIONS_COUNT}\n")
    
    # Verificar cada profesión del mapeo
    print("🔍 Verificando profesiones del mapeo...")
    for profession_name, profession_key in PROFESSION_NAME_MAPPING.items():
        if profession_key not in PROFESSION_CONTEXT_QUESTIONS:
            missing_professions.append((profession_name, profession_key))
            print(f"  ❌ {profession_name} ({profession_key}): FALTAN preguntas de contexto")
        else:
            questions = PROFESSION_CONTEXT_QUESTIONS[profession_key]
            if len(questions) != CONTEXT_QUESTIONS_COUNT:
                incomplete_professions.append((profession_name, profession_key, len(questions)))
                print(f"  ⚠️  {profession_name} ({profession_key}): {len(questions)} preguntas (esperado: {CONTEXT_QUESTIONS_COUNT})")
            else:
                print(f"  ✅ {profession_name} ({profession_key}): {len(questions)} preguntas")
    
    # Verificar profesiones en base de datos que no están en mapeo
    print("\n🔍 Verificando profesiones en base de datos no mapeadas...")
    for profession_id, profession in PROFESSIONS_DATABASE.items():
        profession_name = profession.name
        # Buscar si está en el mapeo
        found = False
        for mapped_name, mapped_key in PROFESSION_NAME_MAPPING.items():
            if mapped_key == profession_id:
                found = True
                break
        
        if not found:
            print(f"  ⚠️  {profession_name} ({profession_id}): No está en PROFESSION_NAME_MAPPING")
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    total_checked = len(all_profession_names)
    total_ok = total_checked - len(missing_professions) - len(incomplete_professions)
    
    print(f"✅ Profesiones correctas: {total_ok}/{total_checked}")
    print(f"❌ Profesiones faltantes: {len(missing_professions)}")
    print(f"⚠️  Profesiones incompletas: {len(incomplete_professions)}")
    
    if missing_professions:
        print("\n❌ PROFESIONES FALTANTES:")
        for name, key in missing_professions:
            print(f"   - {name} ({key})")
    
    if incomplete_professions:
        print("\n⚠️  PROFESIONES INCOMPLETAS:")
        for name, key, count in incomplete_professions:
            print(f"   - {name} ({key}): {count} preguntas (esperado: {CONTEXT_QUESTIONS_COUNT})")
    
    # Detalles de preguntas
    print("\n📋 DETALLES DE PREGUNTAS POR PROFESIÓN:")
    for profession_name, profession_key in sorted(PROFESSION_NAME_MAPPING.items()):
        if profession_key in PROFESSION_CONTEXT_QUESTIONS:
            questions = PROFESSION_CONTEXT_QUESTIONS[profession_key]
            print(f"\n{profession_name} ({profession_key}):")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q[:80]}...")
    
    return len(missing_professions) == 0 and len(incomplete_professions) == 0


if __name__ == "__main__":
    success = verify_context_questions()
    sys.exit(0 if success else 1)

