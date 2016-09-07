package chosen.nlp.lda.conf;

public class LDAParameter {
  public static int K = 20; //topic number
  public static int topNum = 50;
  public static int topicalCoherentTopNum = 50;
  public static float alpha = (float) 0.5; //doc-topic dirichlet prior parameter 
  public static float beta = (float) 0.5;//topic-word dirichlet prior parameter
  public static int iterations = 2501;//Times of iterations
  public static int saveStep = 500 ;//The number of iterations between two saving
  public static int beginSaveIters = 1000;//Begin save model at this iteration 
  public static float pmi_threshold = (float) 0; //the threshold of PMI_v_Sum
  public static int startWeight = 1000;//the start iterations of term weight
  public static int stopWeight = 2500;//the stop iterations of term weight
  public static int  bornedIn = 300;
  public static int saveParameterStep = 1000;
  
  public static double seedParameter = 0.6;
  
  public LDAParameter() {
    // TODO Auto-generated constructor stub
  }
  
  public void getParameter(String parameterPath) {
    //get parameter from file
  }
}
