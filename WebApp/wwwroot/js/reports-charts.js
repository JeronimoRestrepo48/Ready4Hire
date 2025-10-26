// Reports Dashboard Charts
// Inicializa gráficas de reportes usando Chart.js

window.initializePerformanceTrendChart = function(data) {
    const ctx = document.getElementById('performanceTrendChart');
    if (!ctx) return;
    
    // Destruir chart existente si hay
    if (window.performanceTrendChartInstance) {
        window.performanceTrendChartInstance.destroy();
    }
    
    // Crear nueva chart
    window.performanceTrendChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'Score',
                data: data.map(d => d.score),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    padding: 12,
                    borderColor: 'rgba(99, 102, 241, 0.5)',
                    borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `Score: ${context.parsed.y.toFixed(1)}/10`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 5,
                    max: 10,
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(99, 102, 241, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(99, 102, 241, 0.1)'
                    }
                }
            }
        }
    });
};

