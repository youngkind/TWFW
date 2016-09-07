package chosen.nlp.lda.test;

import java.io.IOException;

import chosen.nlp.lda.conf.LDAParameter;
import chosen.nlp.lda.conf.PathConfig;
import chosen.nlp.lda.model.LDA;
import chosen.nlp.lda.model.LdaModel;
import chosen.nlp.lda.model.TWLdaModel;
import chosen.nlp.lda.model.TWLdaTestModel;
import chosen.nlp.lda.model.variables.TWLdaTestVariables;
import chosen.nlp.lda.util.DocSentence;
import chosen.nlp.lda.util.Documents;

public class LdaTest {

  /*
   * 训练样本
   */
  public static void train() throws IOException{
	  DocSentence DocsIn = new DocSentence();
	  String delimiter = "!###";
	  DocsIn.readStructuredDocs(PathConfig.LdaStructuredTrainFilePath, delimiter);
	  
	  System.out.println("Execution of Standard LDA:");
	  TWLdaModel Standard_lda = new TWLdaModel(DocsIn,delimiter);
	  Standard_lda.Train();
	  
	  System.out.println("------------------------------------");
	  System.out.println("Execution of TWLDA:");
	  TWLdaModel TWLda = new TWLdaModel(DocsIn,delimiter);
	  TWLda.TWTrain();
  }
  
  
  /*
   * 测试样本
   */
  public static void test() throws IOException{
    DocSentence DocsTest = new DocSentence();
    DocsTest.read20NewsStructuredDocs("data/LdaTrainSet/2/tcdata/smallTest.txt","data/LdaTrainSet/2/tcdata/label.txt");
    TWLdaTestModel testLda = new TWLdaTestModel(DocsTest, "");
    testLda.TWTrain();
    testLda.saveResult();
    testLda.getPrecision();
  }
  /**
   * @param args
 * @throws IOException 
   */
  public static void main(String[] args) throws IOException {
    // TODO Auto-generated method stub
   
    train();
  }
}
