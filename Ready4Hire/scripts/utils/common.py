"""
Utilidades compartidas para todos los scripts.
"""
import sys
from pathlib import Path


# Path del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """Colores para terminal."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str, width: int = 80):
    """Imprime header con formato."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * width}{Colors.END}\n")


def print_step(step_num: int, total_steps: int, description: str):
    """Imprime paso con formato."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔{'═' * 78}╗{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}║  PASO {step_num}/{total_steps}: {description:<65}║{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * 78}╝{Colors.END}\n")


def print_success(text: str):
    """Imprime éxito."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Imprime error."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Imprime advertencia."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Imprime info."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_section(title: str):
    """Imprime sección."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{title}{Colors.END}")
    print(f"{Colors.HEADER}{'-' * len(title)}{Colors.END}")


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Solicita confirmación al usuario.
    
    Args:
        message: Mensaje a mostrar
        default: Valor por defecto (True=Sí, False=No)
        
    Returns:
        True si el usuario confirma, False en caso contrario
    """
    suffix = "(Y/n)" if default else "(y/N)"
    response = input(f"{message} {suffix}: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes', 'si', 'sí']


def get_paths():
    """Retorna paths importantes del proyecto."""
    return {
        'root': PROJECT_ROOT,
        'app': PROJECT_ROOT / 'app',
        'scripts': PROJECT_ROOT / 'scripts',
        'logs': PROJECT_ROOT / 'logs',
        'data': PROJECT_ROOT / 'data',
        'models': PROJECT_ROOT / 'models',
        'datasets': PROJECT_ROOT / 'app' / 'datasets',
        'tests': PROJECT_ROOT / 'tests',
    }


def ensure_directories():
    """Crea directorios necesarios si no existen."""
    paths = get_paths()
    
    dirs_to_create = [
        paths['logs'],
        paths['data'] / 'training',
        paths['models'],
        paths['datasets'],
        paths['tests'] / 'results',
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)


def print_file_info(file_path: Path, label: str = "Archivo"):
    """Imprime información de un archivo."""
    if file_path.exists():
        size_mb = file_path.stat().st_size / 1024 / 1024
        print_info(f"{label}: {file_path}")
        print(f"      Tamaño: {size_mb:.2f} MB")
    else:
        print_warning(f"{label} no encontrado: {file_path}")
