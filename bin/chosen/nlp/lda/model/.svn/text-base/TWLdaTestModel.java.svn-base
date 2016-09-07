package chosen.nlp.lda.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

public class TWLdaTestModel extends TWLdaModel{
	
	  public TWLdaTestModel(DocSentence DocsIn, String delimiter) {
	    // TODO Auto-generated constructor stubs
		super(DocsIn, delimiter);
	    loadPhi();
	  }

	  public void loadPhi(){
		  String resPath = PathConfig.LdaResultsPath;
		  try {
			BufferedReader structuredReviewsReader = new BufferedReader(
			          new FileReader(new File(resPath+"phi.result")));
			int bufferSize = 1024;
		    char [] buffer = new char[bufferSize];
		    StringBuffer fileData = new StringBuffer();
		    int numRead = 0;
	        while( (numRead = structuredReviewsReader.read(buffer)) != -1) {
	          String readData = String.valueOf(buffer,0,numRead);
	          fileData.append(readData);
	        }
		    String[] lines = fileData.toString().split("\n");
		    
			TWLdaTestVariables trainVariable = new TWLdaTestVariables(lines.length,lines[0].split("\t").length , C);
			for(int k=0; k<lines.length; k++){
				String[] v_item = lines[k].split("\t");
				for(int v=0; v<v_item.length; v++){
					trainVariable.phi[k][v] = Double.valueOf(v_item[v]);
				}
			}
			
			structuredReviewsReader = new BufferedReader(
			          new FileReader(new File(resPath+"vocabulary.result")));
			fileData = new StringBuffer();
			numRead = 0;
			while( (numRead = structuredReviewsReader.read(buffer)) != -1) {
		          String readData = String.valueOf(buffer,0,numRead);
		          fileData.append(readData);
		    }
			lines = fileData.toString().split("\n");
			for(int v=0; v < lines.length; v++){
				trainVariable.vocabulary[v] = lines[v];
			}
			
			structuredReviewsReader = new BufferedReader(
			          new FileReader(new File(resPath+"pkc.result")));
			fileData = new StringBuffer();
			numRead = 0;
			while( (numRead = structuredReviewsReader.read(buffer)) != -1) {
		          String readData = String.valueOf(buffer,0,numRead);
		          fileData.append(readData);
		    }
			lines = fileData.toString().split("\n");
			for(int k=0; k < lines.length; k++){
				String[] item = lines[k].split("\t");
				for(int c=0; c < item.length; c++){
					trainVariable.pkc[k][c] = Double.valueOf(item[c]);
				}
			}
			
			for(int i=0; i < trainVariable.V; i++){
				Integer w_index = trainSet.termToIndexMap.get(trainVariable.vocabulary[i]);
				if (w_index == null){
					continue;
				}
				for(int k = 0; k < K; k++){
					phi[k][w_index] = trainVariable.phi[k][i];
				}
			}
			
			for(int k=0; k < trainVariable.K; k++){
				for(int c=0; c < C; c++){
					pkc[k][c] = trainVariable.pkc[k][c];
				}
			}
			
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	  }
	  

	  protected int TWgibbsSampler(int m, int word, int iter) {
		    // sum of k
		    double topicsSum = 0;
		    double[] topicArray = new double[K];
		    for (int k = 0; k < K; k++) {
		    	if(iter < LDAParameter.startWeight){
		        topicArray[k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha)
		            * phi[k][word];
		    	}
		    	else{
		        topicArray[k] = (twNmkt_tSum[m][k] + alpha)/(twNmkt_tSum_kSum[m] + K * alpha)
		    		  * phi[k][word];
		    	}
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
		    //nmkt[m][currentTopic][word]++;
		    up_nmkt(m, currentTopic, word);
		    TWvariables_add(m,currentTopic,word);
		    return currentTopic;
		  }
	  
	  public void SaveEstimatedParameters() {
		    // updae parameter of phi
		    for (int m = 0; m < M; m++) {
		      for (int k = 0; k < K; k++) {
		        theta[m][k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
		      }
		    }
		  }
	  
	  public void saveResult() throws IOException{
		    //calculation of pkc
		    
		    String resPath = PathConfig.LdaResultsPath;
		    BufferedWriter writer = new BufferedWriter(new FileWriter(resPath + "label.result"));
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
	  
	  public void Save(int iters) throws IOException {
		    // TODO Auto-generated method stub
		    // lda.params lda.phi lda.theta lda.tassign lda.twords
		    // lda.params
		    String resPath = PathConfig.LdaResultsPath;
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
		        + modelName + ".test.theta"));
		    for (int i = 0; i < M; i++) {
		      for (int j = 0; j < K; j++) {
		        writer.write(theta[i][j] + "\t");
		      }
		      writer.write("\n");
		    }
		    writer.close();

		    // lda.phi K*V
		    writer = new BufferedWriter(new FileWriter(resPath + modelName + ".test.phi"));
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
		    writer = new BufferedWriter(new FileWriter(resPath + modelName + ".test.cr"));
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
		        new FileWriter(resPath + modelName + ".test.tassign"));
		    for (int m = 0; m < M; m++) {
		      for (int n = 0; n < doc[m].length; n++) {
		        writer.write(doc[m][n] + ":" + z[m][n] + "\t");
		      }
		      writer.write("\n");
		    }
		    writer.close();

		    // lda.twords phi[][] K*V
		    writer = new BufferedWriter(new FileWriter(resPath + modelName + ".test.twords"));

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

		    writer = new BufferedWriter(new FileWriter(resPath + modelName
		        + ".test.PMIwords"));

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
		          
		    /*
		    Collections.sort(tWordsIndexArray, new TWLdaModel.TwordsComparable(
		    		DBC_v_Sum));
			    for (int t = V-1; t >= 0; t--) {
			      writer.write(trainSet.indexToTermMap.get(tWordsIndexArray.get(t)) + " "
			          + String.format("%.8f", DBC_v_Sum[tWordsIndexArray.get(t)])
			          + "\t");
			          */
			          
		      if (t % 5 == 0)
		        writer.write("\n\t");
		      writer.write("\t");
		    }
		    writer.close();
		  }
	  
}
