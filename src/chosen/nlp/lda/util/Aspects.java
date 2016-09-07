package chosen.nlp.lda.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Aspects {
  public static Map<String, List<String>> aspToSeedList 
    = new HashMap<String, List<String>> (); 
  public static Map <String,String> seedToAspect 
    = new HashMap<String, String> (); 
  
  public Aspects() {
    String asp;
    List <String> seedList = new ArrayList<String>();
    
    /*
    asp = "降息";
    seedList.add("降准_n");
    seedList.add("降准_v");
    seedList.add("股市_n");
    seedList.add("降息_n");
    seedList.add("央行_j");
    seedList.add("银行_n");
    addToMap(asp, seedList);
    */
    
    /*
    asp = "battery";
    seedList.add("battery");
    addToMap(asp, seedList);
    
    seedList = new ArrayList<String>();
    asp = "light";
    seedList.add("light");
    addToMap(asp, seedList);
    
    seedList = new ArrayList<String>();
    asp = "card";
    seedList.add("card");
    addToMap(asp, seedList);
    */
    
    /*
    asp = "picture";
    seedList.add("picture");
    seedList.add("photos");
    seedList.add("image");
    addToMap(asp, seedList);
    
    asp = "len";
    seedList = new ArrayList<String>();
    seedList.add("zoom");
    seedList.add("slr");
    seedList.add("dslr");
    seedList.add("lens");
    addToMap(asp, seedList);
    
    asp = "screen";
    seedList = new ArrayList<String>();
    seedList.add("screen");
    seedList.add("lcd");
    addToMap(asp, seedList);
    
    asp = "price";
    seedList = new ArrayList<String>();
    seedList.add("money");
    seedList.add("price");
    seedList.add("worth");
    addToMap(asp, seedList);
    
    asp = "shooting";
    seedList = new ArrayList<String>();
    seedList.add("shooting");
    seedList.add("shoot");
    seedList.add("shots");
    addToMap(asp, seedList);
    
    asp = "mode";
    seedList = new ArrayList<String>();
    seedList.add("auto");
    seedList.add("mode");
    seedList.add("manual");
    addToMap(asp, seedList);
    
    asp = "shutter";
    seedList = new ArrayList<String>();
    seedList.add("shutter");
    seedList.add("speed");
    addToMap(asp, seedList);
    
    asp = "battery";
    seedList = new ArrayList<String>();
    seedList.add("battery");
    seedList.add("adapter");
    //seedList.add("voltage");
    addToMap(asp, seedList);
    
    
    asp = "light";
    seedList = new ArrayList<String>();
    seedList.add("iso");
    seedList.add("light");
    seedList.add("shot");
    addToMap(asp, seedList);
    
    asp = "speed";
    seedList = new ArrayList<String>();
    seedList.add("shutter");
    seedList.add("speed");
    //seedList.add("different");
    addToMap(asp, seedList);
    
    asp = "size";
    seedList = new ArrayList<String>();
    seedList.add("monitor");
    seedList.add("size");
    addToMap(asp, seedList);
    
    asp = "memory";
    seedList = new ArrayList<String>();
    seedList.add("flash");
    seedList.add("memory");
    seedList.add("volume");
    seedList.add("sd");
    seedList.add("card");
    addToMap(asp, seedList);
    
    asp = "connectivity";
    seedList = new ArrayList<String>();
    seedList.add("connectivity");
    seedList.add("bluetooth");
    //seedList.add("wi-fi");
    addToMap(asp, seedList);
    
    asp = "video";
    seedList = new ArrayList<String>();
    seedList.add("movie");
    seedList.add("video");
    seedList.add("videos");
    addToMap(asp, seedList);
    
    asp = "resolution";
    seedList = new ArrayList<String>();
    seedList.add("display");
    seedList.add("resolution");
    addToMap(asp, seedList);
    
    asp = "port";
    seedList = new ArrayList<String>();
    seedList.add("port");
    seedList.add("usb");
    seedList.add("connector");
    seedList.add("slimport");
    addToMap(asp, seedList);
    
    asp = "system";
    seedList = new ArrayList<String>();
    seedList.add("system");
    seedList.add("operating");
    seedList.add("ios");
    addToMap(asp, seedList);
    
    asp = "sensors";
    seedList = new ArrayList<String>();
    seedList.add("sensors");
    seedList.add("gps");
    seedList.add("compass");
    seedList.add("gyroscope");
    addToMap(asp, seedList);
    */
    
    /*
    asp = "beginner";
    seedList = new ArrayList<String>();
    //seedList.add("user");
    seedList.add("beginner");
    seedList.add("easy");
    addToMap(asp, seedList);
    */
    /*
    asp = "purchase";
    seedList = new ArrayList<String>();
    seedList.add("purchased");
    seedList.add("purchase");
    addToMap(asp, seedList);
    */
  }

  private void addToMap(String asp, List<String> seedList) {
    aspToSeedList.put(asp, seedList);
    for(String seedWord : seedList) {
      seedToAspect.put(seedWord, asp);
    }
  }
  
  public static boolean isSeed( String word ) {
    return (seedToAspect.get(word) != null);
  }
  
  public static String getAspect(String word) {
    return seedToAspect.get(word);
  }
  
}
