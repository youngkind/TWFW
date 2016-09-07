package chosen.nlp.lda.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.CharBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 
 * @author Chosen
 *
 */

public class Documents extends DocBase implements Doc {
	
	public ArrayList<Document> docs; 
	public Map<String,Integer> termToIndexMap;
	public ArrayList<String> indexToTermMap;
	public Map<String,Integer> termCountMap;
	
	public Documents(){
		docs = new ArrayList<Document>();
		termToIndexMap = new HashMap<String, Integer>();
		indexToTermMap = new ArrayList<String>();
		termCountMap = new HashMap<String, Integer>();
	} 
	
	/* (non-Javadoc)
   * @see chosen.nlp.lda.util.Doc#readDocs(java.lang.String)
   */
	@Override
  public void readDocs(String docsPath){
		for(File docFile : new File(docsPath).listFiles()) {
		  //get files form docsPath using listFiles
			Document doc = new Document(docFile.getAbsolutePath(), 
			    termToIndexMap, 
			    indexToTermMap, 
			    termCountMap);
			//get specific file using getAbsolutePath
			docs.add(doc);
		}
	}
	
	/* (non-Javadoc)
   * @see chosen.nlp.lda.util.Doc#readStructuredDocs(java.lang.String, java.lang.String)
   */
	@Override
  @SuppressWarnings("unchecked")
  public void readStructuredDocs(String docsPath ,String delimiter) {
	  //读取结构化文本
	  //下一步要做的是 结构化document 初始化
	  try {
      BufferedReader structuredReviewsReader = new BufferedReader(
          new FileReader(new File(docsPath)));
      StringBuffer fileData = new StringBuffer();
      int bufferSize = 1024;
      char [] buffer = new char[bufferSize];
      int numRead = 0;
      
      try {
        while( (numRead = structuredReviewsReader.read(buffer)) != -1) {
          String readData = String.valueOf(buffer,0,numRead);
          fileData.append(readData);
        }
      } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
      
      structuredReviewsReader.close();
      String [] reviewArray = fileData.toString().split(delimiter);
      List<String> reviewList = new ArrayList<String>();
      Collections.addAll(reviewList, reviewArray);
      
      for(int i = 0;i < reviewList.size();i++) {
        String review = reviewList.get(i);
        //replace punctuation
        review = review.replaceAll("[^a-zA-Z ]", " ");
        Document doc = new Document(
            "doc_" + i,
            review, 
            termToIndexMap, 
            indexToTermMap, 
            termCountMap);
        docs.add(doc);
      }
      
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
	  
	}
	
	public static class Document {	
		private String docName;
		public int[] docWords;
		
		public Document (
		    String docName,
		    String review,
		    Map<String, Integer> termToIndexMap, 
        ArrayList<String> indexToTermMap, 
        Map<String, Integer> termCountMap) {
		  //使用string list 初始化
      this.docName = docName;
      //Read file and initialize word index array
      ArrayList<String> docLines = new ArrayList<String>();
      FileUtil.readLinesFromString(review, docLines); 
      
      mapLineWords(termToIndexMap, indexToTermMap, termCountMap, docLines);
		}
		
		public Document(String docName, 
		    Map<String, Integer> termToIndexMap, 
		    ArrayList<String> indexToTermMap, 
		    Map<String, Integer> termCountMap){
		  
			this.docName = docName;
			//Read file and initialize word index array
      ArrayList<String> docLines = new ArrayList<String>();
      FileUtil.readLines(docName, docLines); 
      
      mapLineWords(termToIndexMap, indexToTermMap, termCountMap, docLines);
		}

    private void mapLineWords(Map<String, Integer> termToIndexMap,
        ArrayList<String> indexToTermMap, 
        Map<String, Integer> termCountMap,
        ArrayList<String> docLines) {
      
      ArrayList<String> words = new ArrayList<String>();
      for(String line : docLines){
        FileUtil.tokenizeAndLowerCase(line, words);
      }
      
      //Remove stop words and noise words
      for(int i = 0; i < words.size(); i++){
        if(Stopwords.isStopword(words.get(i)) || isNoiseWord(words.get(i))){
          words.remove(i);
          i--;
        }
      }
      
      //Transfer word to index
      this.docWords = new int[words.size()];
      for(int i = 0; i < words.size(); i++){
        String word = words.get(i);
        if(!termToIndexMap.containsKey(word)){
          int newIndex = termToIndexMap.size();               //映射表的大小为新的索引值
          termToIndexMap.put(word, newIndex);                 //先对word进行哈希,值为新索引值
          indexToTermMap.add(word);                           //存词入链表,index为新索引值
          termCountMap.put(word, new Integer(1));             //初始化词计数值
          docWords[i] = newIndex;                             //docWords按词序保存索引值
        } else {
          docWords[i] = termToIndexMap.get(word);
          termCountMap.put(word, termCountMap.get(word) + 1); //词计数值 自增1
        }
      }
      words.clear();
    }
		
		public boolean isNoiseWord(String string) {
			// TODO Auto-generated method stub
			string = string.toLowerCase().trim();
			Pattern MY_PATTERN = Pattern.compile(".*[a-zA-Z]+.*");
			Matcher m = MY_PATTERN.matcher(string);
			// filter @xxx and URL
			if(string.matches(".*www\\..*") || string.matches(".*\\.com.*") || 
					string.matches(".*http:.*") || Stopwords.isStopword(string))
				return true;
			if (!m.matches()) {
				return true;
			} else
				return false;
		}
		
	}
}
