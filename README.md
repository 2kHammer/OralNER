# Oral NER

## Allgemeines
Die Anwendung **OralNER** ermöglicht die Anwendung und das Feinanpassen von NER-Modellen auf transkribierte Interviews im .csv-Format des Archiv "Deutsches Gedächtnis".
Sie unterstützt die Frameworks spaCy, Flair und HuggingFace.
OralNER besteht aus einem Backend, das die Funktionalität über eine REST-Schnittstelle zur Verfügung stellt, und einem Frontend, das diese Funktionalität über eine Weboberfläche Anwendern zugänglich macht.
Beides wird in Docker-Containern ausgeführt.
Die Anwendung wurde im Rahmen des Praktikums "Cloud Based Information Extraction" entwickelt.

## Ausführung
Die Anwendung kann auf verschiedene Wege ausgeführt werden, welche im Folgenden erläutert werden.  
Zur Ausführung des Backend-Containers ist es erforderlich, dass die Metadatendatei der Modelle lokal bereitgestellt und in den Container eingebunden wird. In den folgenden Beschreibungen wird davon ausgegangen, dass der Start der Anwendung aus dem Projektverzeichnis erfolgt. Erfolgt der Start aus einem anderen Verzeichnis, muss der Pfad zur Metadatendatei entsprechend angepasst werden.
### Vorgesehene Ausführung
**Das ist der von mir vorgesehene Weg zur Ausführung der Anwendung.**
#### Voraussetzungen
+ Docker
+ Docker-Compose

#### Beschreibung
1. Im Projektverzeichnis `docker compose up -d`bzw. `docker-compose up -d` ausführen
2. Frontend- und Backend-Container werden gestartet
3. Die Webseite ist über `http://localhost:8080`erreichbar

#### Anmerkungen
Der Befehl kann grundsätzlich aus jedem Verzeichnis ausgeführt werden, in dem sich die `docker-compose.yml` befindet.
Damit die Anwendung korrekt funktioniert, muss die Datei `backend/app/store/NER-Models/base_models_metadata.json` aus dem Projektverzeichnis an den gewünschten Speicherort kopiert werden. Anschließend muss der Pfad im `docker-compose.yml` entsprechend angepasst werden. Die modifizierte NER-Modelle werden in einem Docker Volume gespeichert und bleiben somit erhalten.

### Alternative Ausführung 1
Dieser Weg kann verwendet werden, wenn Docker Compose nicht installiert werden kann oder genutzt werden soll. Die Container werden einzeln gestartet.
#### Voraussetzungen
Docker
#### Beschreibung
1. Verzeichnis für die persistente Speicherung der modifizierten Modelle und Metadaten auswählen: `LokalesModellVerzeichnis`
2. Die Datei `backend/app/store/NER-Models/base_models_metadata.json` in das `LokalesModellVerzeichnis` kopieren
3. Backend-Container starten
```bash
docker run -p 5000:5000 \
-v LokalesModellVerzeichnis:/backend/app/store/NER-Models/modified \
-v LokalesModellVerzeichnis/base_models_metadata.json:/backend/app/store/NER-Models/models_metadata.json \
hammeralex2000/oralner:latest
```
5. Ein Konfigurations-Verzeichnis für das Frontend erstellen: `/config`
6. Konfigurations-Datei `/config/config.json` erstellen
   1. Mit der Eigenschaft `backendURL` die URL des Backends angeben 
7. Frontend-Container starten:
`docker run -d -p 8080:80 -v ./config:/etc/nginx/config hammeralex2000/oralner-website:latest`
8. Die Webseite ist über `http://localhost:8080`erreichbar

### Alternative Ausführung 2
Diese Variante sollte nur verwendet werden, falls Docker nicht verfügbar oder nicht installiert ist. Sie wurde nur eingeschränkt getestet und die vollständige Funktionalität kann von den Entwicklern nicht garantiert werden. Dabei wird die Anwendung ohne Container direkt auf dem System ausgeführt.
#### Voraussetzungen
+  Linux
   +  (Die Anwendung wurde unter Linux entwickelt, eine Ausführung der Backends unter Windows sollte nicht möglich sein)
+  Python 3.11 und pip installiert
+  VSCode mit der Erweiterung "Live Server"

#### Beschreibung
1. Python Abhängigkeiten des Backends installieren: `pip install -r ./backend/requirements.txt`
2. Setze den Pythonpfad zum Backend-Verzeichniss: `export PYtHONPATH=PfadZumBackend`
3. Ins Verzeichnis `backend/` wechseln 
4. Backend starten: `python ./app/main.py`
5. In VSCode den Ordner `./frontend/` öffnen
6. Die Datei `./frontend/src/index.html` mit Live-Server öffnen
7. Die Webseite ist über `http://127.0.0.1:5500`erreichbar

## Sonstiges
### Tests
Im Verzeichnis  `./backend/tests` befinden sich die Tests für den Backend-Quellcode. Zur Gewährleistung der korrekten Funktionsweise wurden vor allem Integrationstests mit echten NER-Modellen und Datensätzen eingesetzt. Diese Tests können jedoch nur ausgeführt werden, wenn entsprechende Datensätze und Modelle vorhanden sind. Diese müssen zuvor bereitgestellt werden.
Zudem müssten die Test-Abhängigkeiten aus `./backend/requirements-dev.txt` installiert werden.