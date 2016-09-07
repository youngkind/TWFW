package chosen.nlp.lda.model.variables;

public class TWLdaTestVariables {

	public double[][] phi;
	public double[][] pkc;
	public String[] vocabulary;
	public int K;
	public int V;
	public int C;
	
	public TWLdaTestVariables(int k, int v, int c){
		phi = new double[k][v];
		pkc = new double[k][c];
		vocabulary = new String[v];
		this.K = k;
		this.V = v;
		this.C = c;
	}
}
