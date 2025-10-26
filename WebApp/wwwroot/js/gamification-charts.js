// Gamification Charts with Chart.js

window.initializeGamificationCharts = function() {
    // Destruir gráficas existentes si existen
    Chart.helpers.each(Chart.instances, function(instance) {
        instance.destroy();
    });

    const chartColors = {
        primary: 'rgba(102, 126, 234, 0.8)',
        primaryLight: 'rgba(102, 126, 234, 0.2)',
        secondary: 'rgba(118, 75, 162, 0.8)',
        success: 'rgba(16, 185, 129, 0.8)',
        warning: 'rgba(251, 191, 36, 0.8)',
        danger: 'rgba(239, 68, 68, 0.8)',
        info: 'rgba(59, 130, 246, 0.8)',
        grid: 'rgba(255, 255, 255, 0.1)',
        text: '#9ca3af'
    };

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: chartColors.text,
                    font: {
                        size: 12,
                        family: "'Inter', sans-serif"
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: chartColors.grid
                },
                ticks: {
                    color: chartColors.text
                }
            },
            x: {
                grid: {
                    color: chartColors.grid
                },
                ticks: {
                    color: chartColors.text
                }
            }
        }
    };

    // 1. Weekly Progress Chart (Line Chart)
    const weeklyCtx = document.getElementById('weeklyProgressChart');
    if (weeklyCtx) {
        new Chart(weeklyCtx, {
            type: 'line',
            data: {
                labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                datasets: [{
                    label: 'Puntos ganados',
                    data: [120, 150, 180, 200, 170, 220, 250],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primaryLight,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: chartColors.primary,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    title: {
                        display: false
                    }
                }
            }
        });
    }

    // 2. Games by Type Chart (Doughnut Chart)
    const gamesTypeCtx = document.getElementById('gamesByTypeChart');
    if (gamesTypeCtx) {
        new Chart(gamesTypeCtx, {
            type: 'doughnut',
            data: {
                labels: ['Quiz Rápido', 'Desafío de Código', 'Escenarios', 'Ronda Rápida', 'Constructor'],
                datasets: [{
                    data: [25, 15, 20, 30, 10],
                    backgroundColor: [
                        chartColors.primary,
                        chartColors.success,
                        chartColors.warning,
                        chartColors.info,
                        chartColors.secondary
                    ],
                    borderWidth: 2,
                    borderColor: '#0a0a0a'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: chartColors.text,
                            padding: 15,
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        });
    }

    // 3. Daily Score Chart (Bar Chart)
    const dailyScoreCtx = document.getElementById('dailyScoreChart');
    if (dailyScoreCtx) {
        new Chart(dailyScoreCtx, {
            type: 'bar',
            data: {
                labels: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                datasets: [{
                    label: 'Puntuación',
                    data: [85, 90, 75, 95, 88, 92, 97],
                    backgroundColor: [
                        chartColors.primary,
                        chartColors.primary,
                        chartColors.warning,
                        chartColors.success,
                        chartColors.primary,
                        chartColors.success,
                        chartColors.success
                    ],
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // 4. Response Time Chart (Radar Chart)
    const responseTimeCtx = document.getElementById('responseTimeChart');
    if (responseTimeCtx) {
        new Chart(responseTimeCtx, {
            type: 'radar',
            data: {
                labels: ['Velocidad', 'Precisión', 'Consistencia', 'Dificultad', 'Racha'],
                datasets: [{
                    label: 'Tu Desempeño',
                    data: [85, 90, 75, 80, 95],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primaryLight,
                    pointBackgroundColor: chartColors.primary,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: chartColors.primary
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: chartColors.text
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: chartColors.grid
                        },
                        angleLines: {
                            color: chartColors.grid
                        },
                        pointLabels: {
                            color: chartColors.text,
                            font: {
                                size: 11
                            }
                        },
                        ticks: {
                            color: chartColors.text,
                            backdropColor: 'transparent'
                        }
                    }
                }
            }
        });
    }
};

