import os
import wave
import json
import sys
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

def convert_to_wav(audio_file_path):
    """
    Convierte un archivo de audio a formato WAV con 16kHz y mono.
    Retorna la ruta del archivo WAV convertido.
    """
    file_name, file_extension = os.path.splitext(audio_file_path)
    wav_file_path = file_name + '_converted.wav'

    # Convertir a WAV usando pydub
    try:
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_file_path, format='wav')
        return wav_file_path
    except Exception as e:
        print(f"Error al convertir {audio_file_path} a WAV: {e}")
        return None

def transcribe_audio_vosk(audio_file_path, model, confidence_threshold):
    """
    Transcribe un archivo de audio usando VOSK y marca palabras con baja confianza con '***'.
    Retorna la transcripción como una cadena de texto.
    """
    # Convertir el archivo a WAV con 16kHz mono
    wav_file_path = convert_to_wav(audio_file_path)
    if not wav_file_path:
        return ""
    
    # Abrir el archivo WAV
    try:
        wf = wave.open(wav_file_path, 'rb')
    except Exception as e:
        print(f"Error al abrir {wav_file_path}: {e}")
        return ""
    
    # Verificar formato de audio
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print(f"El archivo {wav_file_path} debe estar en formato WAV con 16kHz, mono")
        wf.close()
        return ""
    
    # Configurar el reconocedor
    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)
    
    transcription = []
    
    # Procesar el archivo de audio
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            words = result.get('result', [])
            for word_info in words:
                word = word_info.get('word', '***')
                conf = word_info.get('conf', 1.0)
                if conf < confidence_threshold:
                    word = '***'
                transcription.append(word)
        # No procesamos resultados parciales para evitar repeticiones
    
    # Procesar el resultado final
    final_result = json.loads(recognizer.FinalResult())
    words = final_result.get('result', [])
    for word_info in words:
        word = word_info.get('word', '***')
        conf = word_info.get('conf', 1.0)
        if conf < confidence_threshold:
            word = '***'
        transcription.append(word)
    
    wf.close()
    
    # Eliminar el archivo WAV convertido
    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)
    
    # Unir las palabras transcritas
    return ' '.join(transcription)

def main():
    # Ruta al modelo VOSK
    model_path = "vosk-model-es-0.42"

    # Verificar si el modelo existe
    if not os.path.exists(model_path):
        print("Modelo no encontrado. Descárgalo desde: https://alphacephei.com/vosk/models")
        sys.exit(1)

    # Cargar el modelo
    print("Cargando el modelo de VOSK...")
    model = Model(model_path)
    print("Modelo cargado correctamente.")

    # Obtener todos los archivos de audio en formatos específicos
    audio_extensions = ['.m4a', '.mp3', '.wav', '.flac','.mp4']
    audio_files = [f for f in os.listdir() if os.path.splitext(f)[1].lower() in audio_extensions]

    if not audio_files:
        print("No se encontraron archivos de audio en el directorio actual.")
        sys.exit(0)

    # Crear carpeta para las transcripciones si no existe
    output_folder = 'transcripciones'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ajustar el umbral de confianza aquí
    confidence_threshold = 0.0  # Cambia este valor según tus necesidades (0.0 - 1.0)

    # Procesar cada archivo de audio
    for audio_file in audio_files:
        print(f"\nTranscribiendo {audio_file}...")
        transcription = transcribe_audio_vosk(audio_file, model, confidence_threshold)
        print(f"Transcripción de {audio_file}:\n{transcription}\n")

        # Guardar la transcripción en un archivo de texto
        output_file = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.txt')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"Transcripción guardada en {output_file}")
        except Exception as e:
            print(f"Error al guardar la transcripción: {e}")

if __name__ == "__main__":
    main()
