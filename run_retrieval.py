import jnius_config

jar_path = r".venv\Lib\site-packages\pyserini\resources\jars\anserini-0.23.0-fatjar.jar"

jnius_config.set_classpath(jar_path)

# Now import the official script
from code.retrievalauggen.get_retrieved_text import main

if __name__ == "__main__":
    main()