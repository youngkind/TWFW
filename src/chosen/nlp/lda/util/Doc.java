package chosen.nlp.lda.util;

public interface Doc {

  public abstract void readDocs(String docsPath);

  public abstract void readStructuredDocs(String docsPath, String delimiter);

}