package chosen.nlp.lda.model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import chosen.nlp.lda.conf.LDAParameter;
import chosen.nlp.lda.conf.PathConfig;
import chosen.nlp.lda.model.variables.PMILabelVariables;
import chosen.nlp.lda.model.variables.TWLdaTestVariables;
import chosen.nlp.lda.util.Aspects;
import chosen.nlp.lda.util.Doc;
import chosen.nlp.lda.util.DocBase;
import chosen.nlp.lda.util.DocSentence;
import chosen.nlp.lda.util.Documents;
import chosen.nlp.lda.util.FileUtil;
import chosen.nlp.lda.util.SIPair;

public class TWLdaModel implements LDA{
	  // private Documents trainSet;
	  protected DocSentence trainSet;
	  // private DocBase trainSet;
	  protected int[][] doc;// word index array
	  protected int V, K, M, C, T;// vocabulary size, topic number, document number ,total word frequent
	  protected int LV; // allocate LV in order to accustom to V
	  protected int[][] z;// topic label array
	  public double alpha; // doc-topic dirichlet prior parameter
	  public double beta[][]; // topic-word dirichlet prior parameter
	  public double betaSum[]; 
	  protected int[][] nmk;// given document m, count times of topic k. M*K
	  protected int[][] nkt;// given topic k, count times of term t. K*V
	  protected int[] nt; // given term t, count is frequent. K
	  //int[][][] nmkt;//given document m,topic k ,count times of term t. M*K*V
	  protected Map<String,Integer> nmkt;
	  // int [][][] nmkt;//given doucument m , count times of topic k & t , used for
	  // termWeight
	  protected int[] nmkSum;// Sum for each row in nmk
	  protected int[] nktSum;// Sum for each row in nkt
	  protected double[][] theta;// Parameters for doc-topic distribution M*K
	  protected double[][] phi;// Parameters for topic-word distribution K*V
	  protected int iterations;// Times of iterations
	  protected int saveStep;// The number of iterations between two saving
	  protected int beginSaveIters;// Begin save model at this iteration
	  protected long timestampt; // Timestamp of the current time  
	  
	  //double[][][] twNmkt;
	  protected Map<String,Double> twNmkt;
	  protected double[][] twNmkt_tSum;
	  protected double[] twNmkt_tSum_kSum;
	  protected double[][] twNkt;
	  protected double[] twNkt_tSum;
	  
	  protected double[] DBC_v_Sum;
	  protected double[][] CR_kv_Sum;
	  protected double[][] IQF_QF_ICF_kv_Sum;
	  protected Map<String,Double> Word_Doc_PMI_vm_Sum;
	  protected double[] VAR_v_Sum;
	  
	  protected double[] pre_DBC_v_Sum;
	  protected double[][] pre_CR_kv_Sum;
	  protected double[][] pre_IQF_QF_ICF_kv_Sum;
	  protected double[][] pre_PMI_DOC_dv_Sum;
	  protected Map<String,Double> pre_Word_Doc_PMI_vm_Sum;
	  protected double[] pre_VAR_v_Sum;
	  
	  
	  protected double[][] pkc;
	  protected TWLdaTestVariables twVariables;

	  public PMILabelVariables plv;
	  
	  protected Map<Integer,Set<Integer>> word_doc_map; // use a word index can get the set of document index which include this word
	  protected Map<Integer,Set> word_topic_map; //user a word index can get the set of topic which this word has.
	  
	  protected Map<String,Double> TFIDF; //each word in document has a tfidf-value ,length:V
	  protected Map<String,Double> pre_TFIDF;
	  protected double[] TF; // term frequency of terms
	  
	  /*
	   * public LdaModel(Documents DocsIn) { // TODO Auto-generated constructor stub
	   * trainSet = new Documents(); trainSet.readDocs(PathConfig.LdaTrainSetPath);
	   * getParameter(); this.Initialize(DocsIn); }
	   * 
	   * public LdaModel(Documents DocsIn,String delimiter) { // TODO Auto-generated
	   * constructor stub trainSet = DocsIn; getParameter();
	   * this.Initialize(DocsIn); }
	   */
	  public TWLdaModel(DocSentence DocsIn, String delimiter) {
	    // TODO Auto-generated constructor stubs
	    trainSet = DocsIn;
	    getParameter();
	    this.Initialize(DocsIn);
	  }

	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#Train()
	   */

	  @Override
	  public void Train() {
	    if (iterations < saveStep + beginSaveIters) {
	      System.err
	          .println("Error: the number of iterations should be larger than "
	              + (saveStep + beginSaveIters));
	      System.exit(0);
	    }
	    for (int i = 0; i < iterations; i++) {
	      if (i % 100 == 0 && i != 0){
	    	// update parameter
	    	SaveEstimatedParameters();
	    	printIterate(i,"standardLDA");
	      }
	      if (i == iterations - 1) {
	        // Saving the model
	        System.out.println("Saving model at iteration " + i + " ... ");
	        
	        // Secondly print model variables
	        try {
	          Save(i, trainSet);
	        } catch (IOException e) {
	          // TODO Auto-generated catch block
	          e.printStackTrace();
	        }
	      }
	      // gibbs sampling
	      gibbsSample();
	    }
	  }

	  public void calTopicWordPMI() {
		  
		  	//calculate PMI
		    for (int v = 0; v < V; v++) {
		    	String v_word = trainSet.indexToTermMap.get(v);
		    	double p_tSum = (double) trainSet.termCountMap.get(v_word);
		    	for (int k = 0; k < K; k++){
		    		double p_kt = nkt[k][v]/p_tSum;
		    		plv.PMI[k][v] = p_kt*Math.log(1+p_kt);
		    		if(Double.isNaN(plv.PMI[k][v]))
		    			plv.PMI[k][v] = 0;
		    	}
		    	double PMI_Sum = 0;
		    	for (int k = 0; k < K; k++){
		    		PMI_Sum += plv.PMI[k][v];
		    	}
		    	plv.PMI_v_Sum[v] = 1 + PMI_Sum/Math.log(K);
		      }
		    
		  //calculate BDC,CR,IQF_QF_ICF
		  for(int v=0;v<V;v++){
			  double phi_kSum = 0;
			  double bdcSum = 0;
			  String v_word = trainSet.indexToTermMap.get(v);
			  int tSum = trainSet.termCountMap.get(v_word);
			  
			  for (int k = 0; k < K; k++){
				  //phi_kSum += phi[k][v];
				  phi_kSum += nkt[k][v]/(double)nktSum[k];
			  }
			  
			  for(int k=0; k < K; k++){
				  //bdcSum += (phi[k][v]/phi_kSum)*Math.log(phi[k][v]/phi_kSum);
				  if(phi_kSum != 0){
					  double total = (nkt[k][v]/(double)nktSum[k])/phi_kSum;
					  if (total != 0 ){
						  bdcSum += (total)*Math.log(total);
					  }
				  }
				  //CR_kv_Sum[k][v] = Math.log(2+(nkt[k][v]*V/(nktSum[k]*tSum)));
				  pre_CR_kv_Sum[k][v] = CR_kv_Sum[k][v];
				  CR_kv_Sum[k][v] = 0.1 + (nkt[k][v]*V/((double)nktSum[k]*tSum));
				  //calculate IQF_QF_ICF_kv_Sum
				  pre_IQF_QF_ICF_kv_Sum[k][v] = IQF_QF_ICF_kv_Sum[k][v];
				  IQF_QF_ICF_kv_Sum[k][v] = Math.log10(T/(double)(nt[v])) * Math.log10(nkt[k][v] + 1) * Math.log10(K/(double)word_topic_map.get(v).size() + 1.0);
				  //CR_kv_Sum[k][v] = 1;
			  }
			  pre_DBC_v_Sum[v] = DBC_v_Sum[v];
			  DBC_v_Sum[v] = 1 + bdcSum/Math.log(K);
			  //DBC_v_Sum[v] = 1;
		  }
		  
		  //calculate VAR
		  double[] phi_kv_sum = new double[K]; 
		  for (int k = 0; k < K; k++){
			  double sum = 0;
			  for (int v = 0; v < V; v++){
				  sum += phi[k][v];
			  }
			  phi_kv_sum[k] = sum;
		  }
		  
		  for (int v1 = 0; v1 < V; v1++){
			  double[] varianceArray = new double[K];
			  double average = 0;
			  double sqrsum = 0;
			  for (int k = 0; k < K; k++){
				  varianceArray[k] = phi[k][v1] / phi_kv_sum[k];
				  average += varianceArray[k];
				  sqrsum += varianceArray[k]*varianceArray[k];
			  }
			  average = average / K;
			  pre_VAR_v_Sum[v1] = VAR_v_Sum[v1];
			  VAR_v_Sum[v1] = (sqrsum - K * average * average) / K;
		  }
		  
		  //calculate TFIDF,PMI
		  for (int m = 0; m < M; m++){
			  int NN = doc[m].length;
			  for ( int nn = 0; nn < NN; nn++){
				  int v = doc[m][nn];
				  int N = doc[m].length;
				  String index = m + "_" + v;
				  int vCount = 0;
				  for (int n = 0; n < N; n++){
					  if (doc[m][n] == v){
						  vCount += 1;
					  }
				  }
				  double tf = vCount/(double)N;
				  double df = (double)M/word_doc_map.get(v).size();
				  pre_TFIDF.put(index, TFIDF.get(index));
				  TFIDF.put(index, tf*Math.log10(df));
				  
				  //calculate PMI
				  //double pxd = vCount / (double)NN;
				  double pxd = vCount;
				  //double px = nt[doc[m][nn]] / (double)T;
				  double px = nt[doc[m][nn]];
				  double pmi = -Math.log(pxd/px);
				  if (pmi < 0){
					  pmi = 0;
				  }
				  String wdIndex = doc[m][nn] + "_" + m;
				  pre_Word_Doc_PMI_vm_Sum.put(wdIndex, Word_Doc_PMI_vm_Sum.get(wdIndex));
				  Word_Doc_PMI_vm_Sum.put(wdIndex, pmi);
			  }
		  }
		  
		  //change the counting
		  for (int m = 0; m < M; m++) {
		      int N = doc[m].length;
		      for (int n = 0; n < N; n++) {
		        // Sample from p(z_i|z_-i, w)
		        int currentTopic = z[m][n];
		        int word = doc[m][n];
		        double pmi = getWeight(word,currentTopic,m,true);
		        //double pmi = pre_DBC_v_Sum[word]*pre_CR_kv_Sum[currentTopic][word];
				//twNmkt[m][k][t] -= pmi;
				twNmkt_tSum[m][currentTopic] -= pmi;
				twNmkt_tSum_kSum[m] -= pmi;
				twNkt[currentTopic][word] -= pmi;
				twNkt_tSum[currentTopic] -= pmi;
				pmi = getWeight(word,currentTopic,m,false);
				//pmi = DBC_v_Sum[word] * CR_kv_Sum[currentTopic][word];
				twNmkt_tSum[m][currentTopic] += pmi;
				twNmkt_tSum_kSum[m] += pmi;
				twNkt[currentTopic][word] += pmi;
				twNkt_tSum[currentTopic] += pmi;
		      }
		    }
	  }
	  
	  public void TermWeightingInit(){
		  for(int v=0; v<V; v++){
			  for(int k=0; k<K; k++){
				  pre_CR_kv_Sum[k][v] = 1;
				  CR_kv_Sum[k][v] = 1;
				  IQF_QF_ICF_kv_Sum[k][v] = 1;
				  pre_IQF_QF_ICF_kv_Sum[k][v] = 1;
			  }
			  pre_DBC_v_Sum[v] = 1;
			  DBC_v_Sum[v] = 1;
			  plv.PMI_v_Sum[v] = 1;
			  pre_VAR_v_Sum[v] = 1;
			  VAR_v_Sum[v] = 1;
		  }
		  TFIDF = new HashMap<String,Double>();
		  pre_TFIDF = new HashMap<String,Double>();
		  for(int m=0; m<M; m++){
			  int N = doc[m].length;
			  for(int n=0; n<N; n++){
				  TFIDF.put(m+"_"+doc[m][n], 1.0);
				  pre_TFIDF.put(m+"_"+doc[m][n], 1.0);
				  if(!Word_Doc_PMI_vm_Sum.containsKey(doc[m][n]+"_"+m)){
					  Word_Doc_PMI_vm_Sum.put(doc[m][n] + "_" + m, 1.0);
					  pre_Word_Doc_PMI_vm_Sum.put(doc[m][n] + "_" + m, 1.0);
				  }
			  }
		  }
	  }

	  private void gibbsSample() {
	    // Use Gibbs Sampling to update z[][]
	    for (int m = 0; m < M; m++) {
	      int N = doc[m].length;
	      for (int n = 0; n < N; n++) {
	        // Sample from p(z_i|z_-i, w)
	        int newTopic = Sample(m, n);
	        z[m][n] = newTopic;
	      }
	    }
	  }
	  

	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#getParameter()
	   */
	  @Override
	  public void getParameter() {
	    K = LDAParameter.K;
	    // alpha = 1.0 / K;
	    // beta = 1.0 / V;
	    iterations = LDAParameter.iterations;
	    saveStep = LDAParameter.saveStep;
	    beginSaveIters = LDAParameter.beginSaveIters;
	  }

	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#Initialize(chosen.nlp.lda.util.Documents)
	   */

	  @Override
	  public void Initialize(Documents docSet) {
	    M = docSet.docs.size();
	    V = docSet.termToIndexMap.size();
	    C = trainSet.docLabel.length;
	    T = 0;
	    // autolly set super-parameter
	    initVariables();

	    doc = new int[M][];
	    for (int m = 0; m < M; m++) {
	      int N = docSet.docs.get(m).docWords.length;
	      doc[m] = new int[N];
	      for (int n = 0; n < N; n++) {
	        doc[m][n] = docSet.docs.get(m).docWords[n];
	        T ++;
	      }
	    }
	    sampleInit();
	  }

	  @SuppressWarnings("null")
	  public void Initialize(DocSentence docSet) {
	    M = docSet.docs.size();
	    V = docSet.termToIndexMap.size();
	    C = trainSet.docLabel.length;
	    T = 0;
	    // init time stamp
	    timestampt = System.currentTimeMillis();
	    // autolly set super-parameter
	    initVariables();

	    // reset beta & betaSum for seed words
	    
	    int seedTopic = 0;
	    Collection<List<String>> seedwordCollect = Aspects.aspToSeedList.values();
	    for (List<String> sameAspect : seedwordCollect) {
	      if (sameAspect != null || sameAspect.isEmpty()) {
	        double seedParameter = 0.3;
	        resetBeta(sameAspect, seedTopic, seedParameter);
	        seedTopic++;
	      }
	    }
	    

	    doc = new int[M][];
	    word_doc_map = new HashMap<Integer,Set<Integer>>(); 
	    for (int m = 0; m < M; m++) {
	      int N = docSet.docs.get(m).docWords.length;
	      doc[m] = new int[N];
	      for (int n = 0; n < N; n++) {
	        doc[m][n] = docSet.docs.get(m).docWords[n];
	        if (!word_doc_map.containsKey(doc[m][n])){
	        	word_doc_map.put(doc[m][n], new HashSet<Integer>());
	        }
	        word_doc_map.get(doc[m][n]).add(m);  // generate the word_doc_map. each word relate to a set of document index which include this word
	        T ++;
	      }
	    }
	    sampleInitFromTopic(seedTopic);
	    
	    
	    seedTopic = 0;
	    for (List<String> sameAspect : seedwordCollect) {
	      if (sameAspect != null) {
	        for (int m = 0; m < this.M; m++) {
	          sampleInit(sameAspect, trainSet.docs.get(m).seedDSPair,
	              trainSet.docs.get(m).SIPairIndexMap, m, seedTopic);
	        }
	      }
	      seedTopic++;
	    }
	    
	    TF = new double[V];
	    for (int v = 0; v < V; v++){
	    	int wordCount = 0;
	    	int wordCountSum = 0;
	    	for(int k = 0; k < K; k++){
	    		wordCount += nkt[k][v];
	    		wordCountSum += nktSum[k];
		  }
		  double tf = wordCount/(double)wordCountSum;
		  TF[v] = tf;
	    }
	  }

	  private void initVariables() {
	    alpha = 1.0 / K;

	    beta = new double[K][V];
	    betaSum = new double[K];
	    for (int i = 0; i < K; i++) {
	      for (int j = 0; j < V; j++) {
	        beta[i][j] = 1.0 / V;
	      }
	      betaSum[i] = 1.0;
	    }
	    nmk = new int[M][K];
	    nkt = new int[K][V];
	    nt = new int[V];
	    //nmkt = new int [M][K][V];
	    nmkt = new HashMap<String,Integer>();
	    nmkSum = new int[M];
	    nktSum = new int[K];
	    theta = new double[M][K];
	    phi = new double[K][V];
	    
	    plv = new PMILabelVariables();
	    plv.PMI = new double [K][V];
	    plv.PMI_v_Sum = new double [V];
	    
	    //twNmkt = new double[M][K][V];
	    twNmkt_tSum = new double[M][K];
	    twNmkt_tSum_kSum = new double[M];
	    twNkt = new double[K][V];
	    twNkt_tSum = new double[K];
	    DBC_v_Sum = new double[V];
	    CR_kv_Sum = new double[K][V];
	    IQF_QF_ICF_kv_Sum = new double[K][V];
	    VAR_v_Sum = new double[V];
	    pre_VAR_v_Sum = new double[V];
	    pre_DBC_v_Sum = new double[V];
	    pre_CR_kv_Sum = new double[K][V];
	    pre_IQF_QF_ICF_kv_Sum = new double[K][V];
	    Word_Doc_PMI_vm_Sum = new HashMap<String,Double>();
	    pre_Word_Doc_PMI_vm_Sum = new HashMap<String,Double>();
	    pkc = new double[K][C];
	  }

	  private void sampleInit() {
	    // sample topic ramdomly
	    z = new int[M][];
	    for (int m = 0; m < M; m++) {
	      int N = doc[m].length;
	      z[m] = new int[N];
	      for (int n = 0; n < N; n++) {
	        int word = doc[m][n];
	        int topic = (int) (Math.random() * K);
	        z[m][n] = topic;
	        nmk[m][topic]++;
	        //nmkt[m][topic][word] ++;
	        //up_nmkt(m, topic, word);
	        nkt[topic][word]++;
	        nktSum[topic]++;
	        nt[word] ++;
	      }
	      nmkSum[m] = N;
	    }
	  }
	  
	  private void sampleInitFromTopic(int fromTopic) {
      // sample topic ramdomly
      z = new int[M][];
      word_topic_map = new HashMap<Integer,Set>();
      for (int m = 0; m < M; m++) {
        int N = doc[m].length;
        z[m] = new int[N];
        for (int n = 0; n < N; n++) {
          int word = doc[m][n];
          int topic = fromTopic + (int) (Math.random() * (K - fromTopic));
          z[m][n] = topic;
          nmk[m][topic]++;
          //nmkt[m][topic][word] ++;
          //up_nmkt(m, topic, word);
          nkt[topic][word]++;
          nktSum[topic]++;
          nt[word] ++;
          if(!word_topic_map.containsKey(word)){
	        	word_topic_map.put(word, new HashSet<Integer>());
	        }
	        word_topic_map.get(word).add(topic);
        }
        nmkSum[m] = N;
      }
    }

	  @SuppressWarnings("unused")
	  private void sampleInit(List<String> sameAspect,
	      Map<String, List<SIPair>> seedDSPair,
	      Map<String, Integer> SIPairIndexMap, int m, int topic) {
	    int former;
	    int word;
	    for (String seedWord : sameAspect) {
	      // for each word in sameAspect,get its position and constrain .
	      List<SIPair> siPairs = seedDSPair.get(seedWord);
	      if (siPairs == null)
	        continue;
	      for (SIPair siPair : siPairs) {
	        // resample
	        int n = SIPairIndexMap.get(siPair.toString());
	        word = doc[m][n];
	        former = z[m][n];
	        nmk[m][former]--;
	        nkt[former][word]--;
	        nktSum[former]--;

	        z[m][n] = topic;
	        nmk[m][topic]++;
	        nkt[topic][word]++;
	        nktSum[topic]++;
	      }
	    }
	  }

	  @SuppressWarnings("unused")
	  private void resetBeta(List<String> sameAspect, int topic,
	      double seedParameter) {
	    for (String seedWord : sameAspect) {
	      if (!trainSet.termToIndexMap.containsKey(seedWord)) {
	        continue;
	      }
	      int word = this.trainSet.termToIndexMap.get(seedWord);
	      for (int i = 0; i < this.K; i++) {
	        beta[i][word] = 0;
	        betaSum[i] -= 1 / this.V;
	      }
	      beta[topic][word] = seedParameter;
	      betaSum[topic] += seedParameter;
	    }
	  }

	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#SaveEstimatedParameters()
	   */
	  public void SaveTWEstimatedParameters() {
	    // update parameter of theta
	    for (int k = 0; k < K; k++) {
	      for (int t = 0; t < V; t++) {
    		  //phi[k][t] = (nkt[k][t] + beta[k][t]) / (nktSum[k] + betaSum[k]);
	    	  phi[k][t] = (twNkt[k][t] + beta[k][t]) / (twNkt_tSum[k] + betaSum[k]);
	      }
	    }
	    // updae parameter of phi
	    for (int m = 0; m < M; m++) {
	      for (int k = 0; k < K; k++) {
	        theta[m][k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
	      }
	    }
	  }

	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#SaveEstimatedParameters()
	   */
	  @Override
	  public void SaveEstimatedParameters() {
	    // update parameter of theta
	    for (int k = 0; k < K; k++) {
	      for (int t = 0; t < V; t++) {
	        phi[k][t] = (nkt[k][t] + beta[k][t]) / (nktSum[k] + betaSum[k]);
	      }
	    }
	    // updae parameter of phi
	    for (int m = 0; m < M; m++) {
	      for (int k = 0; k < K; k++) {
	        theta[m][k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
	      }
	    }
	  }
	  
	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#Save(int, chosen.nlp.lda.util.Documents)
	   */
	  @Override
	  public void Save(int iters) throws IOException {
	    // TODO Auto-generated method stub
	    // lda.params lda.phi lda.theta lda.tassign lda.twords
	    // lda.params
	    String resPath = PathConfig.LdaResultsPath + timestampt + "/";
	    //FileUtil.mkdir(new File(resPath));
	    String modelName = "lda_" + iters;
	    ArrayList<String> lines = new ArrayList<String>();
	    lines.add("alpha = " + alpha);
	    lines.add("beta = " + beta);
	    lines.add("topicNum = " + K);
	    lines.add("docNum = " + M);
	    lines.add("termNum = " + V);
	    lines.add("iterations = " + iterations);
	    lines.add("saveStep = " + saveStep);
	    lines.add("beginSaveIters = " + beginSaveIters);
	    FileUtil.writeLines(resPath + modelName + ".params", lines);

	    
	    // lda.theta M*K
	    BufferedWriter writer = new BufferedWriter(new FileWriter(resPath
	        + modelName + ".train.theta"));
	    /*
	    for (int i = 0; i < M; i++) {
	      for (int j = 0; j < K; j++) {
	        writer.write(theta[i][j] + "\t");
	      }
	      writer.write("\n");
	    }
	    writer.close();

	    // lda.phi K*V
	    writer = new BufferedWriter(new FileWriter(resPath + modelName + ".train.phi"));
	    for (int i=0; i < V; i++){
	    	String word = trainSet.indexToTermMap.get(i);
	    	writer.write(word);
	    	for (int j=0; j < K; j++){
	    		writer.write(","+phi[j][i]+"\t");
	    	}
	    	writer.write("\n");
	    }
	    writer.close();
	    
	    // lda.cr K*V
	    writer = new BufferedWriter(new FileWriter(resPath + modelName + ".train.cr"));
	    for (int v=0; v < V; v++){
	    	String word = trainSet.indexToTermMap.get(v);
	    	writer.write(word);
	    	for (int k=0; k < K; k++){
	    		writer.write(","+CR_kv_Sum[k][v]);
	    	}
	    	writer.write("\n");
	    }

	    // lda.tassign
	    writer = new BufferedWriter(
	        new FileWriter(resPath + modelName + ".train.tassign"));
	    for (int m = 0; m < M; m++) {
	      for (int n = 0; n < doc[m].length; n++) {
	        writer.write(doc[m][n] + ":" + z[m][n] + "\t");
	      }
	      writer.write("\n");
	    }
	    writer.close();
      */
	    
	    // lda.twords phi[][] K*V
	    writer = new BufferedWriter(new FileWriter(resPath + modelName + ".train.twords"));

	    int topNum = LDAParameter.topNum; // Find the top 20 topic words in each
	                                      // topic

	    for (int i = 0; i < K; i++) {
	      List<Integer> tWordsIndexArray = new ArrayList<Integer>();
	      for (int j = 0; j < V; j++) {
	        tWordsIndexArray.add(new Integer(j));
	      }
	      // Construct index of phi[i] sorted by its value
	      // which means that index = tWordsIndexArray.get(0) , phi[i][index] is the
	      // biggest value
	      Collections.sort(tWordsIndexArray, new TWLdaModel.TwordsComparable(phi[i]));
	      writer.write("\n\n\n" + "topic " + i + "\t:\t" + nktSum[i] + "\t");
	      for (int t = 0; t < topNum; t++) {

	        writer
	            .write(trainSet.indexToTermMap.get(tWordsIndexArray.get(t)) + " "
	                + String.format("%.8f", phi[i][tWordsIndexArray.get(t)]) 
	                +" "+ nkt[i][tWordsIndexArray.get(t)] + "\t");

	        if (t % 5 == 0)
	          writer.write("\n\t");
	      }
	      writer.write("\n");
	    }
	    writer.close();
      
      
	    //输出按照PMI从大到小排序的词序列
	    writer = new BufferedWriter(new FileWriter(resPath + modelName
	        + ".train.PMIwords"));

	    List<Integer> tWordsIndexArray = new ArrayList<Integer>();
	    for (int j = 0; j < V; j++) {
	      tWordsIndexArray.add(new Integer(j));
	    }
	    // Construct index of phi[i] sorted by its value
	    // which means that index = tWordsIndexArray.get(0) , phi[i][index] is the
	    // biggest value
	    
	    Collections.sort(tWordsIndexArray, new TWLdaModel.TwordsComparable(
	        plv.PMI_v_Sum));
	    for (int t = V-1; t >= 0; t--) {
	      writer.write(trainSet.indexToTermMap.get(tWordsIndexArray.get(t)) + " "
	          + String.format("%.8f", plv.PMI_v_Sum[tWordsIndexArray.get(t)])
	          + "\t");
	    
	    
	    //Collections.sort(tWordsIndexArray, new TWLdaModel.TwordsComparable(
	    //		DBC_v_Sum));
		  // for (int t = V-1; t >= 0; t--) {
		  //    writer.write(trainSet.indexToTermMap.get(tWordsIndexArray.get(t)) + " "
		  //        + String.format("%.8f", DBC_v_Sum[tWordsIndexArray.get(t)])
		  //        + "\t");
		          
		          
	      if (t % 5 == 0)
	        writer.write("\n\t");
	      writer.write("\t");
	    }
	    writer.close();
	    
	    //输出当前文档分类结果，输出的形式：word_topic，词与词之间逗号间隔
	    writer = new BufferedWriter(new FileWriter(resPath + modelName
          + ".train.DocTopic"));
	    for (int m = 0; m < M; m++){
	      int N = doc[m].length;
	      for (int n = 0; n < N; n++){
	        int topic = z[m][n];
	        int word = doc[m][n];
	        writer.write(word+"_"+topic);
	        if ( n != N-1){
	          writer.write(",");
	        }
	      }
	      writer.write("\n");
	    }
	    writer.close();
	    
	    //输出文档词典，输出格式为：编号,词语
	    writer = new BufferedWriter(new FileWriter(resPath + modelName
          + ".train.Doc"));
	    for (int t = 0; t < V; t++) {
        writer.write(t+","+trainSet.indexToTermMap.get(t)+"\n");
	    }
	    writer.close();
	    
	    //存储每个topic的topical coherent值
	    writer = new BufferedWriter(new FileWriter(resPath + modelName
	            + ".train.TopicCoherent"));
	    for (int k = 0; k < K; k++){
	    	writer.write("topic " + k + ":" +Topical_coherent(k) + "\n");
	    }
	    writer.close();
	  }

	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#Sample(int, int)
	   */
	  @Override
	  public int Sample(int m, int n) {
	    int word = doc[m][n];
	    // Remove topic label for w_{m,n}
	    int formerTopic = z[m][n];
	    nmk[m][formerTopic]--;
	    nkt[formerTopic][word]--;
	    nmkSum[m]--;
	    nktSum[formerTopic]--;

	    return gibbsSampler(m, word);
	  }
	  

	  private int gibbsSampler(int m, int word) {
	    // sum of k
	    double topicsSum = 0;
	    double[] topicArray = new double[K];
	    for (int k = 0; k < K; k++) {
	      topicArray[k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha)
	          * (nkt[k][word] + beta[k][word]) / (nktSum[k] + betaSum[k]);
	      topicsSum += topicArray[k];
	    }
	    //
	    int currentTopic = 0;
	    double topicCursor = Math.random() * topicsSum;
	    for (int k = 0; k < K; k++) {
	      topicCursor -= topicArray[k];
	      if (topicCursor <= 0) {
	        currentTopic = k;
	        break;
	      }
	    }
	    nmk[m][currentTopic]++;
	    nkt[currentTopic][word]++;
	    nmkSum[m]++;
	    nktSum[currentTopic]++;
	    return currentTopic;
	  }
	  
	  public int TWSample(int m, int n, int iter) {
		    int word = doc[m][n];
		    // Remove topic label for w_{m,n}
		    int formerTopic = z[m][n];
		    
		    // if word is seedword, the topic will not change
		    /*
		    String s_word = trainSet.indexToTermMap.get(word);
		    Collection<List<String>> seedwordCollect = Aspects.aspToSeedList.values();
		    for (List<String> sameAspect : seedwordCollect) {
		      for (String aspect : sameAspect){
		        if(s_word.contains(aspect)){
		          return formerTopic;
		        }
		      }
		    }
		    */
		    
		    nmk[m][formerTopic]--;
		    nkt[formerTopic][word]--;
		    nmkSum[m]--;
		    nktSum[formerTopic]--;
		    //nmkt[m][formerTopic][word] --;
		    //down_nmkt(m, formerTopic, word);
		    TWvariables_sub(m,formerTopic,word);

		    return TWgibbsSampler(m, word, iter);
		  }
	  
	  protected int TWgibbsSampler(int m, int word, int iter) {
		    // sum of k
		    double topicsSum = 0;
		    double[] topicArray = new double[K];
		    if(iter < LDAParameter.startWeight){
		    	double nmk_sum = nmkSum[m] + K * alpha;
		    	for (int k = 0; k < K; k++) {
		    		topicArray[k] = (nmk[m][k] + alpha) / (nmk_sum)
				            * (nkt[k][word] + beta[k][word]) / (nktSum[k] + betaSum[k]);
		    		topicsSum += topicArray[k];
		    	}
		    }
		    else{
		      
		      //用Term Weighting来进行采样 
		      
		    	double nmkt_sum = twNmkt_tSum_kSum[m] + K * alpha;
		    	for (int k = 0; k < K; k++) {
		    		topicArray[k] = (twNmkt_tSum[m][k] + alpha)/(nmkt_sum)
				    		  //* (twNkt[k][word] + beta[k][word]) / (twNkt_tSum[k] + betaSum[k]);
		    					* (nkt[k][word] + beta[k][word]) / (nktSum[k] + betaSum[k]);
		    		topicsSum += topicArray[k];
		    	}
		    	
		    	
		      //使用普通的方法
		      /*
		      double nmk_sum = nmkSum[m] + K * alpha;
	          for (int k = 0; k < K; k++) {
	            topicArray[k] = (nmk[m][k] + alpha) / (nmk_sum)
	                    * (nkt[k][word] + beta[k][word]) / (nktSum[k] + betaSum[k]);
	            topicsSum += topicArray[k];
	          }
	          */
          
		    }
		    
		    
		    //
		    int currentTopic = 0;
		    double topicCursor = Math.random() * topicsSum;
		    for (int k = 0; k < K; k++) {
		      topicCursor -= topicArray[k];
		      if (topicCursor <= 0) {
		        currentTopic = k;
		        break;
		      }
		    }
		    nmk[m][currentTopic]++;
		    nkt[currentTopic][word]++;
		    nmkSum[m]++;
		    nktSum[currentTopic]++;
		    //nmkt[m][currentTopic][word]++;
		    //up_nmkt(m, currentTopic, word);
		    TWvariables_add(m,currentTopic,word);
		    return currentTopic;
		  }
	  
	  public void TWTrain() throws IOException {
		    if (iterations < saveStep + beginSaveIters) {
		      System.err
		          .println("Error: the number of iterations should be larger than "
		              + (saveStep + beginSaveIters));
		      System.exit(0);
		    }
		    TermWeightingInit();
		    TWVariables_init();
		    for (int i = 0; i < iterations; i++) {
		      
		      //if ((i >= beginSaveIters) && (((i - beginSaveIters) % saveStep) == 0)) {
		      //最后一步才保存模型
		      if (i == iterations - 1) {
		        // Saving the model
		        System.out.println("Saving model at iteration " + i + " ... ");
		        
		        // Firstly update parameters
		        //SaveEstimatedParameters();
		        // Secondly print model variables
		        try {
		          Save(i, trainSet);
		        } catch (IOException e) {
		          // TODO Auto-generated catch block
		          e.printStackTrace();
		        }
		      }
		      if((i >= LDAParameter.startWeight) && i <= LDAParameter.stopWeight  && (((i - LDAParameter.startWeight) % LDAParameter.saveParameterStep) == 0)){
		    	  calTopicWordPMI();
		    	  
				  //TWVariables_init();
		      }
		      if (i % 100 == 0 && i != 0){
		    	  SaveTWEstimatedParameters();
		    	  printIterate(i,"TWLDA");
		      }
		      // gibbs sampling
		      TWgibbsSample(i);
		    }
		  }
	  
	  private void TWgibbsSample(int iter) {
		    // Use Gibbs Sampling to update z[][]
		    for (int m = 0; m < M; m++) {
		      int N = doc[m].length;
		      for (int n = 0; n < N; n++) {
		        // Sample from p(z_i|z_-i, w)
		        int newTopic = TWSample(m, n, iter);
		        z[m][n] = newTopic;
		      }
		    }
		  }
	  
	  public void TWVariables_init(){
		  //twNmkt = new double[M][K][V];
		  twNmkt_tSum = new double[M][K];
		  twNmkt_tSum_kSum = new double[M];
		  twNkt = new double[K][V];
		  twNkt_tSum = new double[K];
		  
		  /*
		  for(int m=0;m<M;m++){
			  for(int k=0;k<K;k++){
				  for(int t=0;t<V;t++){
					  //twNmkt[m][k][t] = plv.PMI_v_Sum[t]*nmkt[m][k][t];
					  //twNmkt[m][k][t] = DBC_v_Sum[t]*CR_kv_Sum[k][t]*get_nmkt(m, k, t);
					  //twNmkt_tSum[m][k] += twNmkt[m][k][t];
					  twNmkt_tSum[m][k] += DBC_v_Sum[t]*CR_kv_Sum[k][t]*get_nmkt(m, k, t);
				  }
				  twNmkt_tSum_kSum[m] += twNmkt_tSum[m][k];
			  }
		  }
		  */
		  
		  for (int m = 0; m < M; m++) {
		      int N = doc[m].length;
		      for (int n = 0; n < N; n++) {
		        // Sample from p(z_i|z_-i, w)
		        int k = z[m][n];
		        int t = doc[m][n];
		        //double weight = DBC_v_Sum[t]*CR_kv_Sum[k][t];
		        double weight = getWeight(t,k,m,false);
		        twNmkt_tSum[m][k] += weight;
		        twNmkt_tSum_kSum[m] += weight;
		        twNkt[k][t] += weight;
		        twNkt_tSum[k] += weight;
		      }
		  }
		  
		  /*
		  for(int k=0;k<K;k++){
			  for(int t=0;t<V;t++){
				  //twNkt[k][t] = plv.PMI_v_Sum[t]*nkt[k][t];
				  twNkt[k][t] = DBC_v_Sum[t]*CR_kv_Sum[k][t]*nkt[k][t];
				  twNkt_tSum[k] += twNkt[k][t];
			  }
		  }
		  */
	  }
	  
	  public void TWvariables_sub(int m,int k,int t){
		  //double pmi = plv.PMI_v_Sum[t];
		  //Double pmi = DBC_v_Sum[t]*CR_kv_Sum[k][t];
		  Double pmi = getWeight(t,k,m,false);
		  //twNmkt[m][k][t] -= pmi;
		  twNmkt_tSum[m][k] -= pmi;
		  twNmkt_tSum_kSum[m] -= pmi;
		  twNkt[k][t] -= pmi;
		  twNkt_tSum[k] -= pmi;
	  }
	  
	  public void TWvariables_add(int m,int k,int t){
		  //double pmi = plv.PMI_v_Sum[t];
		  //double pmi = DBC_v_Sum[t]*CR_kv_Sum[k][t];
		  double pmi = getWeight(t,k,m,false);
		  //twNmkt[m][k][t] += pmi;
		  twNmkt_tSum[m][k] += pmi;
		  twNmkt_tSum_kSum[m] += pmi;
		  twNkt[k][t] += pmi;
		  twNkt_tSum[k] += pmi;
	  }
	  /*
	   * (non-Javadoc)
	   * 
	   * @see chosen.nlp.lda.model.LDA#SampleAll()
	   */
	  @Override
	  public void SampleAll() {
	    for (int m = 0; m < M; m++) {
	      int N = doc[m].length;
	      for (int n = 0; n < N; n++) {
	        z[m][n] = this.Sample(m, n);
	      }
	    }
	  }
	  
	  protected void up_nmkt(int m, int k, int t){
		  Integer num = nmkt.get(m+"_"+k+"_"+t);
		  if (num != null){
			  nmkt.put(m+"_"+k+"_"+t, num+1);
		  }
		  else{
			  nmkt.put(m+"_"+k+"_"+t, 1);
		  }
	  }
	  
	  private void down_nmkt(int m, int k, int t){
		  nmkt.put(m+"_"+k+"_"+t, nmkt.get(m+"_"+k+"_"+t)-1);
	  }
	  
	  private int get_nmkt(int m, int k, int t){
		  Integer num = nmkt.get(m+"_"+k+"_"+t);
		  if(num != null){
			  return num;
		  }
		  else{
			  return 0;
		  }
	  }
	  
	  private boolean is_nmkt_null(int m, int k, int t){
		  if(nmkt.containsKey(m+"_"+k+"_"+t)){
			  return true;
		  }
		  else{
			  return false;
		  }
	  }
	  
	  public void saveResult() throws IOException{
		    //calculation of pkc
		    calPkc();
		    
		    String resPath = PathConfig.LdaResultsPath;
		    BufferedWriter writer = new BufferedWriter(new FileWriter(resPath + "phi.result"));
		    for (int i = 0; i < K; i++) {
		      for (int j = 0; j < V; j++) {
		        writer.write(phi[i][j] + "\t");
		      }
		      writer.write("\n");
		    }
		    writer.close();
		    
		    writer = new BufferedWriter(new FileWriter(resPath + "vocabulary.result"));
		    for (int i = 0; i < V; i++){
		    	writer.write(trainSet.indexToTermMap.get(i)+"\n");
		    }
		    writer.close();
		    
		    writer = new BufferedWriter(new FileWriter(resPath + "pkc.result"));
		    for (int k = 0; k < K; k++){
		    	writer.write(pkc[k][0]+"");
		    	for (int c = 1; c < C; c++){
		    		writer.write("\t"+pkc[k][c]);
		    	}
		    	writer.write("\n");
		    }
		    writer.close();
		    
		    writer = new BufferedWriter(new FileWriter(resPath + "label.result"));
		    for (int m = 0; m < M; m++){
		    	double max_k_sum = 0;
		    	int max_c_index = -1;
		    	for(int c=0; c < C; c++){
		    		double k_sum = 0;
			    	for(int k=0; k < K; k++){
			    		k_sum += theta[m][k]*pkc[k][c];
			    	}
			    	if(k_sum > max_k_sum){
			    		max_k_sum = k_sum;
			    		max_c_index = c;
			    	}
		    	}
		    	writer.write(m+"\t"+max_c_index+"\t"+trainSet.docLabel[m]+"\n");
		    }
		    writer.close();
	  }
	  
	  //打印出当前的迭代数，topical_coherent值，将其写入文件中
	  private void printIterate(int i,String model){
		  String resPath = PathConfig.LdaResultsPath + timestampt + "/";
		  FileUtil.mkdir(new File(resPath));
		  BufferedWriter writer;
		  try {
			  writer = new BufferedWriter(new FileWriter(resPath + model + "_iteration.csv",true));
			  double average_tc = 0;
			  for ( int k = 0; k < K; k++){
				  average_tc += Topical_coherent(k);
			  }
			  writer.write(i + "," + average_tc/K + "\n");
			  writer.close();
			  System.out.println("Iteration " + i + " topical coherent:" + average_tc/K);
		  } catch (IOException e) {
			  // TODO Auto-generated catch block
			  e.printStackTrace();
		  }
	  }
	  
	  //得到结果的精确度
	  public void getPrecision(){
		  int c_count = 0;
		  for (int m = 0; m < M; m++){
	    	double max_k_sum = 0;
	    	int max_c_index = -1;
	    	for(int c=0; c < C; c++){
	    		double k_sum = 0;
		    	for(int k=0; k < K; k++){
		    		k_sum += theta[m][k]*pkc[k][c];
		    	}
		    	if(k_sum > max_k_sum){
		    		max_k_sum = k_sum;
		    		max_c_index = c;
		    	}
	    	}
	    	if(max_c_index == trainSet.docLabel[m]){
	    		c_count ++;
	    	}
	    }
	    System.out.print((float)c_count/M);
	  }
	  
	  public void calPkc(){
		  for(int k=0; k<K; k++){
			  double c_sum = 0;
			  for(int m=0; m<M; m++){
				  c_sum += theta[m][k];
			  }
			  for(int c=0; c<C; c++){
				double cd_sum = 0; 
				for(int m=0; m<M; m++){
					if(trainSet.docLabel[m] == c){
						cd_sum += theta[m][k];
					}
				}
				pkc[k][c] = cd_sum/c_sum;
			  }
		  }
	  }
	  
	  public TWLdaTestVariables getTrainResult(){
		  calPkc();
		  
		  twVariables = new TWLdaTestVariables(K, V, C);
		  twVariables.phi = phi;
		  twVariables.pkc = pkc;
		  for(int v = 0; v < V; v++){
			  twVariables.vocabulary[v] = trainSet.indexToTermMap.get(v);
		  }
		  return twVariables;
	  }
	  
	  /*
	  * calculate the topical coherent of topic k 
	  */
	  public double Topical_coherent(int k){
		  
		  List<Integer> tWordsIndexArray = new ArrayList<Integer>();
		  int topNum = LDAParameter.topicalCoherentTopNum;
		  for (int j = 0; j < V; j++) {
			  tWordsIndexArray.add(new Integer(j));
	      }
		  Collections.sort(tWordsIndexArray, new TWLdaModel.TwordsComparable(phi[k]));
		  int ct = 0;
		  for (int m = 2; m <= topNum; m++){
			  for (int l = 1; l <= m - 1; l++){
				  Set<Integer> result = new HashSet<Integer>();
				  result.addAll(word_doc_map.get(tWordsIndexArray.get(l)));
				  int dt = result.size();
				  result.retainAll(word_doc_map.get(tWordsIndexArray.get(m)));
				  int dmt = result.size();
				  ct += Math.log10((double)(dmt+1)/dt);
			  }
		  }
		  return ct;
	  }
	  
	  
	  /*
	   * t : index of word
	   * k : topic
	   * m : index of document
	   * pre = true : get pre weighting value
	   * pre = False : get current weighting value
	   */
	  private double getWeight(int t, int k, int m, boolean pre){
		  /*
		   * dbc
		   */
		  //return 1;
		  
		  if(pre == true){
			  return pre_DBC_v_Sum[t];
		  }
		  else{
			  return DBC_v_Sum[t];
		  }
		  
		  /*
		   * tf_bdc
		   */
		  /*
		  if(pre == true){
			  return pre_DBC_v_Sum[t]*getLogistValue(TF[t], 0.001, 0.02);
		  }
		  else{
			  //double test = DBC_v_Sum[t]*getLogistValue(TF[t], 0.001, 0.02);
			  return DBC_v_Sum[t]*getLogistValue(TF[t], 0.001, 0.02);
		  }
		  */
		  /*
		   * cr
		   */
		  /*
		  if(pre == true){
			  return pre_CR_kv_Sum[k][t];
		  }
		  else{
			  return CR_kv_Sum[k][t];
		  }
		  */
		  /*
		   * tf-idf
		   */
		  /*
		  if(pre == true){
			  return pre_TFIDF.get(m+"_"+t);
		  }
		  else{
			  return TFIDF.get(m+"_"+t);
		  }
		  */
		  /*
		   * S model of dbc
		   */
		  /*
		  if(pre == true){
			  return getLogistValue(pre_DBC_v_Sum[t]);
		  }
		  else{
			  return getLogistValue(DBC_v_Sum[t]);
		  }
		  */
		  /*
		   * iqf*qf*icf
		   */
		  /*
		  if(pre == true){
			  return pre_IQF_QF_ICF_kv_Sum[k][t];
		  }
		  else{
			  return IQF_QF_ICF_kv_Sum[k][t];
		  }
		  */
		  /*
		   * PMI
		   */
		  /*
		  String wdIndex = t + "_" + m;
		  if(pre == true){
			  return pre_Word_Doc_PMI_vm_Sum.get(wdIndex);
		  }
		  else{
			  return Word_Doc_PMI_vm_Sum.get(wdIndex);
		  }
		  */
		  /*
		  if(pre == true){
			  return pre_VAR_v_Sum[t];
		  }
		  else{
			  return VAR_v_Sum[t];
		  }
		  */
	  }
	  
	  /*
	   * t : index of word
	   * m : index of document
	   */
	  private double getTFIDFValue (int t, int m){
		  int wordCount = 0;
		  int wordCountSum = 0;
		  for(int k = 0; k < K; k++){
			  wordCount += nkt[k][t];
			  wordCountSum += nktSum[k];
		  }
		  double tf = wordCount/(double)wordCountSum;
		  double df = (double)M/word_doc_map.get(t).size();
		  
		  return tf*Math.log10(df);
	  }
	  
	  /*
	   * logist funciton : change function to S shape
	   */
	  private double getLogistValue(double x, double tension, double center){
		  return 1.0/(1.0 + Math.pow(Math.E, -(x-tension)/center));
	  }
	  
	  /*
	   * logist funciton : change function to S shape
	   */
	  private double getLogistValue(double x){
		  return 1.0/(1.0 + Math.pow(Math.E, -(x-0.5)/0.1));
	  }
	  
	  public class TwordsComparable implements Comparator<Integer> {

	    public double[] sortProb; // Store probability of each word in topic k

	    public TwordsComparable(double[] sortProb) {
	      this.sortProb = sortProb;
	    }

	    @Override
	    public int compare(Integer o1, Integer o2) {
	      // TODO Auto-generated method stub
	      // Sort topic word index according to the probability of each word in
	      // topic k
	      if (sortProb[o1] > sortProb[o2])
	        return -1;
	      else if (sortProb[o1] < sortProb[o2])
	        return 1;
	      else
	        return 0;
	    }
	  }

	  @Override
	  public void Initialize(Doc docSet) {
	    // TODO Auto-generated method stub

	  }

	  @Override
	  public void Save(int iters, Documents docSet) throws IOException {
	    // TODO Auto-generated method stub
	    Save(iters);
	  }

	  @Override
	  public void Save(int iters, DocBase docSet) throws IOException {
	    // TODO Auto-generated method stub
	    Save(iters);
	  }
}
