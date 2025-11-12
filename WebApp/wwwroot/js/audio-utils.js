/**
 * Audio Utilities for STT/TTS Integration
 * Ready4Hire - WebApp
 */

let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;

/**
 * Inicializa el MediaRecorder para grabación de audio
 * @returns {Promise<MediaRecorder>}
 */
window.initializeMediaRecorder = async function() {
    try {
        // Solicitar acceso al micrófono
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            } 
        });

        // Crear MediaRecorder
        mediaRecorder = new MediaRecorder(audioStream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        // Resetear chunks
        audioChunks = [];

        // Event listeners
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = function() {
            console.log('🎤 Grabación detenida');
        };

        console.log('🎤 MediaRecorder inicializado');
        return mediaRecorder;
    } catch (error) {
        console.error('❌ Error al inicializar MediaRecorder:', error);
        throw new Error('No se pudo acceder al micrófono. Verifica los permisos.');
    }
};

/**
 * Inicia la grabación de audio
 * @param {MediaRecorder} recorder 
 */
window.startRecording = function(recorder) {
    try {
        if (recorder && recorder.state === 'inactive') {
            audioChunks = [];
            recorder.start();
            console.log('🔴 Grabación iniciada');
        }
    } catch (error) {
        console.error('❌ Error al iniciar grabación:', error);
        throw error;
    }
};

/**
 * Detiene la grabación y retorna el audio blob
 * @param {MediaRecorder} recorder 
 * @returns {Promise<Blob>}
 */
window.stopRecording = function(recorder) {
    return new Promise((resolve, reject) => {
        try {
            if (recorder && recorder.state === 'recording') {
                recorder.onstop = function() {
                    try {
                        // Crear blob del audio grabado
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        
                        // Limpiar stream
                        if (audioStream) {
                            audioStream.getTracks().forEach(track => track.stop());
                        }
                        
                        console.log('⏹️ Grabación detenida, blob creado:', audioBlob.size, 'bytes');
                        resolve(audioBlob);
                    } catch (error) {
                        reject(error);
                    }
                };
                
                recorder.stop();
            } else {
                reject(new Error('Recorder no está en estado recording'));
            }
        } catch (error) {
            console.error('❌ Error al detener grabación:', error);
            reject(error);
        }
    });
};

/**
 * Convierte un Blob a array de bytes
 * @param {Blob} blob 
 * @returns {Promise<Uint8Array>}
 */
window.blobToBytes = async function(blob) {
    try {
        const arrayBuffer = await blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        console.log('📁 Blob convertido a bytes:', bytes.length, 'bytes');
        return bytes;
    } catch (error) {
        console.error('❌ Error al convertir blob:', error);
        throw error;
    }
};

/**
 * Crea un elemento audio desde array de bytes
 * @param {Uint8Array} audioBytes 
 * @returns {HTMLAudioElement}
 */
window.createAudioFromBytes = function(audioBytes) {
    try {
        // Crear blob desde bytes (el backend retorna mp3 por defecto)
        const audioBlob = new Blob([audioBytes], { type: 'audio/mp3' });
        
        // Crear URL objeto
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Crear elemento audio
        const audioElement = new Audio(audioUrl);
        audioElement.preload = 'auto';
        
        console.log('🔊 Elemento audio creado desde bytes');
        return audioElement;
    } catch (error) {
        console.error('❌ Error al crear audio:', error);
        throw error;
    }
};

/**
 * Reproduce un elemento audio
 * @param {HTMLAudioElement} audioElement 
 */
window.playAudio = async function(audioElement) {
    try {
        if (audioElement) {
            await audioElement.play();
            console.log('▶️ Reproduciendo audio');
        }
    } catch (error) {
        console.error('❌ Error al reproducir audio:', error);
        throw error;
    }
};

/**
 * Detiene la reproducción de audio
 * @param {HTMLAudioElement} audioElement 
 */
window.stopAudio = function(audioElement) {
    try {
        if (audioElement) {
            audioElement.pause();
            audioElement.currentTime = 0;
            console.log('⏹️ Audio detenido');
        }
    } catch (error) {
        console.error('❌ Error al detener audio:', error);
        throw error;
    }
};

/**
 * Configura callback para cuando termine la reproducción
 * @param {HTMLAudioElement} audioElement 
 * @param {DotNetObjectReference} dotNetRef 
 */
window.setupAudioEndCallback = function(audioElement, dotNetRef) {
    try {
        if (audioElement && dotNetRef) {
            audioElement.onended = function() {
                console.log('🔚 Audio terminado, llamando callback');
                dotNetRef.invokeMethodAsync('OnAudioEnded');
            };
        }
    } catch (error) {
        console.error('❌ Error al configurar callback:', error);
        throw error;
    }
};

/**
 * Verifica si el navegador soporta MediaRecorder
 * @returns {boolean}
 */
window.isMediaRecorderSupported = function() {
    return 'MediaRecorder' in window && 'getUserMedia' in navigator.mediaDevices;
};

/**
 * Solicita permisos de micrófono
 * @returns {Promise<boolean>}
 */
window.requestMicrophonePermission = async function() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop()); // Cerrar stream inmediatamente
        return true;
    } catch (error) {
        console.error('❌ Permisos de micrófono denegados:', error);
        return false;
    }
};

// Limpiar recursos al cerrar la página
window.addEventListener('beforeunload', function() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
});

console.log('🎵 Audio utilities loaded successfully');
