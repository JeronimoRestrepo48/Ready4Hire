#!/usr/bin/env python3
"""
PASO 1.1: Generación de Datos de Demostración
Crea evaluaciones sintéticas para probar el sistema.
"""
import sys
from pathlib import Path

# Agregar paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.common import *
from app.infrastructure.ml.training_data_collector import TrainingDataCollector
import json
import random
from datetime import datetime, timedelta


class DemoDataGenerator:
    """Generador de datos de demostración."""
    
    def __init__(self):
        self.tech_questions = [
            "Explica la diferencia entre let, const y var en JavaScript",
            "¿Qué es el Virtual DOM en React?",
            "Describe el patrón MVC en desarrollo web",
            "¿Cómo funciona el event loop en Node.js?",
            "Explica los principios SOLID con ejemplos",
            "¿Qué son las promesas en JavaScript?",
            "Describe REST vs GraphQL",
            "¿Qué es dependency injection?",
            "Explica closures en JavaScript",
            "¿Cómo funcionan los hooks en React?",
            "Describe el patrón Observer",
            "¿Qué es JWT y cómo funciona?",
            "Explica async/await en JavaScript",
            "¿Qué son los microservicios?",
            "Describe Docker y containerización",
        ]
        
        self.soft_questions = [
            "Cuéntame sobre un proyecto desafiante que completaste",
            "¿Cómo manejas conflictos en el equipo?",
            "Describe una situación donde tuviste que aprender algo nuevo rápidamente",
            "¿Cómo priorizas tareas con múltiples deadlines?",
            "Cuéntame sobre un error que cometiste y qué aprendiste",
            "¿Cómo das y recibes feedback?",
            "Describe tu estilo de trabajo en equipo",
            "¿Cómo manejas el estrés y la presión?",
        ]
    
    def generate_good_answer(self) -> str:
        """Genera respuesta de buena calidad."""
        templates = [
            "En mi experiencia, esto se relaciona con varios conceptos importantes. Primero, es fundamental entender que la arquitectura debe ser escalable. He implementado soluciones similares en producción con resultados excelentes, mejorando el rendimiento en un 40%.",
            "Esta es una pregunta interesante que toca varios aspectos fundamentales. En mis proyectos anteriores, he trabajado extensivamente con estas tecnologías. La clave está en balancear performance con mantenibilidad, algo que logré implementando patrones de diseño apropiados.",
            "Basándome en mi experiencia profesional de 5 años, puedo decir que la mejor práctica aquí es considerar múltiples factores. He liderado equipos implementando soluciones similares con gran éxito, reduciendo bugs en un 60% y mejorando la satisfacción del cliente.",
        ]
        return random.choice(templates)
    
    def generate_bad_answer(self) -> str:
        """Genera respuesta de baja calidad."""
        templates = [
            "Eh, no estoy muy seguro. Creo que tiene algo que ver con programación.",
            "Mmm, sí, lo he escuchado pero no recuerdo exactamente cómo funciona.",
            "No lo he usado mucho, pero debe ser algo parecido a otras cosas que he visto.",
            "No sé, nunca trabajé con eso específicamente. Tal vez podría aprenderlo.",
        ]
        return random.choice(templates)
    
    def generate(self, num_samples: int = 200):
        """Genera evaluaciones."""
        print_section("Generando evaluaciones de demostración")
        
        evaluations = []
        
        for i in range(num_samples):
            is_tech = random.random() < 0.6
            is_good = random.random() < 0.7
            
            question = random.choice(self.tech_questions if is_tech else self.soft_questions)
            answer = self.generate_good_answer() if is_good else self.generate_bad_answer()
            score = random.uniform(7.5, 10.0) if is_good else random.uniform(3.0, 6.5)
            
            evaluation = {
                "question": question,
                "answer": answer,
                "score": round(score, 1),
                "category": "technical" if is_tech else "soft_skills",
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                "metadata": {
                    "is_demo": True,
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "confidence": round(random.uniform(0.6, 1.0), 2)
                }
            }
            
            evaluations.append(evaluation)
            
            if (i + 1) % 50 == 0:
                print_info(f"Generadas {i + 1}/{num_samples} evaluaciones")
        
        return evaluations
    
    def save(self, evaluations, output_path: Path):
        """Guarda evaluaciones en audit_log.jsonl."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for eval_data in evaluations:
                log_entry = {
                    "timestamp": eval_data["timestamp"],
                    "event_type": "evaluation_completed",
                    "data": {
                        "question": eval_data["question"],
                        "answer": eval_data["answer"],
                        "score": eval_data["score"],
                        "category": eval_data["category"],
                        "metadata": eval_data["metadata"]
                    }
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Genera datos de demostración")
    parser.add_argument("--num-samples", type=int, default=200, help="Número de evaluaciones")
    parser.add_argument("--output", default="logs/audit_log.jsonl", help="Archivo de salida")
    
    args = parser.parse_args()
    
    print_header("PASO 1.1: GENERACIÓN DE DATOS DEMO")
    
    paths = get_paths()
    output_path = paths['root'] / args.output
    
    try:
        # Generar
        generator = DemoDataGenerator()
        evaluations = generator.generate(args.num_samples)
        
        print_success(f"Generadas {len(evaluations)} evaluaciones")
        
        # Guardar
        print_section("Guardando evaluaciones")
        saved_path = generator.save(evaluations, output_path)
        
        print_success(f"Guardado en: {saved_path}")
        
        # Estadísticas
        tech_count = sum(1 for e in evaluations if e["category"] == "technical")
        soft_count = len(evaluations) - tech_count
        avg_score = sum(e["score"] for e in evaluations) / len(evaluations)
        
        print_section("Estadísticas")
        print_info(f"Total: {len(evaluations)} evaluaciones")
        print_info(f"Técnicas: {tech_count} ({tech_count/len(evaluations)*100:.1f}%)")
        print_info(f"Soft Skills: {soft_count} ({soft_count/len(evaluations)*100:.1f}%)")
        print_info(f"Score promedio: {avg_score:.2f}/10")
        
        print_header("✅ PASO 1.1 COMPLETADO")
        print_info("Próximo paso: python3 scripts/1_data/step2_convert_to_training.py")
        
        return 0
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
