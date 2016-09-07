package chosen.nlp.lda.util;
import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;

import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.morph.*;

public class Wordnet_Stemmer {
	
	public WordnetStemmer wordStem;
	public POS[] pos = {POS.NOUN,POS.ADJECTIVE,POS.VERB,POS.ADVERB};
	
	public Wordnet_Stemmer(){
		String wnhome = System.getenv("WNHOME");
		String path = wnhome + File.separator + "dict";
		URL url = null;
		try
		{
			url = new URL("file", null, path);
		} catch (MalformedURLException e)
		{
			e.printStackTrace();
		}
		IDictionary dict = new Dictionary(url);
		try
		{
			dict.open();
		} catch (Exception e)
		{
			e.printStackTrace();
		}
		wordStem = new WordnetStemmer(dict);
	}

	public String stemString(String s) {
		if(s == " " || s.isEmpty()){
			return s;
		}
		String taget_word = s;
		for(int i=0;i<4;i++){
			List<String> ls = wordStem.findStems(s, pos[i]);
			if(ls.size()>=1){
				taget_word = ls.get(0);
				break;
			}
		}
		return taget_word;
	}
	
	public static void main (String[] args) {
		
		Wordnet_Stemmer ws = new Wordnet_Stemmer();
		String taget_word = ws.stemString("pictures.");
		
		System.out.print(taget_word);
	}
}
