////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                                                                            //
//                            FQH_2_Layers  version 1.00                      //
//                                                                            //
//                        Copyright (C) 2012-2014 Bin Xu                      //
//                                                                            //
//                                                                            //
//    This program studies the 2 layer fractional quantum Hall system on      //
//    a torus. We use exact diagonalization in "m" space and implement the    //
//    momentum conservation in ky.                                            //
//                                                                            //
//    V1.0 (09/08/2014): A completed version that studies the projected       //
//    Coulomb interaction with interlayer hopping terms. It is still buggy.   //
//    V2.0 (16/08/2014): A totally finished version. This specific program    //
//    is designed for large systems in which only one sector will be stored.  //
//    -- This does not work due to the hugh number of matrix elements.        //
//    -- Possible solution 1: use translational symmetry to greatly reduce    //
//       the number of matrix elements.                                       //
//       Pro: fast, nearly no additional cost                                 //
//       Con: the symmetry is not guarenteed                                  //
//    -- Possible solution 2: compute the matrix elements on the fly, or at   //
//       least partially on the fly.                                          //
//       Pro: saves memory, and it's a more common practice                   //
//       Con: it takes long time to calculate the matrix again and again      //
//    -- Possible solution 3: Use scalapack to run full MPI code.             //
//       Pro: industrially mature                                             //
//       Con: you've got to give me lots of free nodes!!                      //
//    -- Possible solution 4: Store matrix elements on the disk               //
//       Pro: saves cpu                                                       //
//       Con: it takes long time to read from disks                           //
//                                                                            //
//    WARNING: variables related to the matrix elements need to be written in //
//    type llong rather than int in this case. We are indeed touching the     //
//    boundary of programming limits... But Haldane's Lanczos is still OK as  //
//    it only deals with the vector which is smaller than the maximum size    //
//    of an integer.                                                          //
//    V3.0 (16/08/2014): I realized that the matrix elements are stored with  //
//    redundancies (symmetric matrix, and anti-symmetrization of matrix       //
//    entries). By eliminating the redundencies, the matrix can be stored in  //
//    one node of a cluster.                                                  //
//                                                                            //
//                        last modification : 16/08/2014                      //
//                                                                            //
//    This program is free software; you can redistribute it and/or modify    //
//    it under the terms of the GNU General Public License as published by    //
//    the Free Software Foundation; either version 2 of the License, or       //
//    (at your option) any later version.                                     //
//                                                                            //
//    This program is distributed in the hope that it will be useful,         //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of          //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           //
//    GNU General Public License for more details.                            //
//                                                                            //
//    You should have received a copy of the GNU General Public License       //
//    along with this program; if not, write to the Free Software             //
//    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <bitset>
#include <algorithm>
#include <lanczos.h>
using namespace std;
#include "mkl_lapacke.h"
const double pi = 3.14159;

const int MaxOrbital = 40;

const int StateIdShift = 100;

const int MaxLapackSize = 5000;

const double SmallDouble = 0.000000001;

const double SmallMomentum = 0.0001;

////////////////////////////////////////////////////////////////////////////////
//Structual structures
////////////////////////////////////////////////////////////////////////////////
struct Hamiltonian
{
    unsigned norb, nele, mrange;
    int lanczosNE;
    double t, a, b;
    double CoulombForm[MaxOrbital][MaxOrbital][MaxOrbital][MaxOrbital];
    char interaction;
    int sector;
    long long int matrixsize;
    
};

Hamiltonian ham; //The global object that provides all complicated parameters

struct Orbital
{
    char layer;
    int m;
    
    bool operator== (const Orbital &rhs) const
    { return m == rhs.m && layer == rhs.layer; }
    Orbital operator+ (const Orbital &rhs) const
    {
        Orbital temp( (m+rhs.m)%ham.mrange, layer);
        if (layer != rhs.layer)
        {
            cout<<"Orbital addition error: cannot add two orbitals in different layers"<<endl;
            abort();
        }
        return temp;
    }
    Orbital(){}
    Orbital(int o_m, char o_layer) { m=o_m, layer = o_layer;}
};

struct Orbital_hasher
{
    size_t operator()(const Orbital& orb) const
    {
        return (hash<int>()(orb.m)^(hash<char>()(orb.layer)<<1));
    }
};

struct OrbPair
{
    Orbital orb1, orb2;
    char layer;
    int m;
    
    OrbPair(){}
    OrbPair(const Orbital& orba, const Orbital& orbb)
    {
        orb1 = orba, orb2 = orbb; m=((orb1.m+orb2.m)%ham.mrange);
        if (orb1.layer != orb2.layer)
        {
            cout<<"OrbPair error: cannot form pairs in different layers"<<endl;
            abort();
        }
        layer = orb1.layer;
    }
};

typedef bitset<MaxOrbital> CompactState;

struct State
{
    CompactState cstate;
    unsigned state_id;
    State(){state_id = 0;}
    State(CompactState& o_cstate){cstate = o_cstate; state_id = 0;}
};

////////////////////////////////////////////////////////////////////////////////
// Matrix and diagonalization related
////////////////////////////////////////////////////////////////////////////////
struct MatEle
{
    int bra, ket;
    double amplitude;
    MatEle()
    {
        bra = 0, ket = 0, amplitude = 0;
    }
};

typedef vector<MatEle> Matrix;

struct bra_ket
{
    int bra, ket;
    bra_ket(){bra = 0; ket = 0;}
    bra_ket(int o_bra, int o_ket){bra=o_bra, ket = o_ket; }
    bool operator== (const bra_ket &rhs) const
    {
        return (bra == rhs.bra && ket == rhs.ket);
    }
};

struct bra_ket_hasher
{
    size_t operator()(const bra_ket & bk) const
    {
        return (hash<int>()(bk.bra) ^ (hash<int>()(bk.ket) <<1));
    }
};

typedef unordered_map<bra_ket, double, bra_ket_hasher > DupMatrix;
//DupMatrix.first = bra and  ket, DupMatrix.second = amplitude


struct diag_return
{
    vector<double> eigenvalues;
    int sector_indicator;
};
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Relations
////////////////////////////////////////////////////////////////////////////////
//Orbital -> orbital id
typedef unordered_map<Orbital, char, Orbital_hasher> OrbMap;
//Orbital property -> orbital pairs with this prop
typedef unordered_map<Orbital, vector<OrbPair>, Orbital_hasher> PairGroup;
//Momentum -> states with this momentum
//typedef unordered_map<int, vector<CompactState> > cStateMap;
typedef vector<CompactState> cStateMap;
//State occupation -> state id
typedef unordered_map<unsigned long long, unsigned int> ReferenceMap;



////////////////////////////////////////////////////////////////////////////////
// Output functions
////////////////////////////////////////////////////////////////////////////////
ostream& operator<<(ostream& os, Orbital& orb)
{
    os << "m = " << orb.m << "\t Layer: " << (int)orb.layer;
    return os;
}
ostream& operator<<(ostream& os, const Orbital& orb)
{
    os << "m = " << orb.m << "\t Layer: " << (int)orb.layer;
    return os;
}

ostream& operator<<(ostream& os, OrbPair& pair)
{
    os << "Orbital m1 = " << pair.orb1.m << "\tOrbital m2 = "<< pair.orb2.m << "\t Layer: "<<(int) pair.orb1.layer;
    return os;
}
ostream& operator<<(ostream & os, MatEle & mat_ele)
{
    os << "<" << mat_ele.bra << "|H|" << mat_ele.ket << "> = "\
    << mat_ele.amplitude;
    return os;
}

ostream& operator<<(ostream & os, State state)
{
    os<< state.state_id<<"\t";
    for (int i = 0; i < ham.norb; i++) os<<state.cstate[i];
    os<<endl;
    return os;
}

void printFullMatEle(MatEle& mat_ele, vector<State>& statelist)
{
    CompactState tempket = statelist[mat_ele.ket].cstate, tempbra = statelist[mat_ele.bra].cstate;
    for (int i = 0; i < ham.norb; i+=2) {
        cout<< tempket[i]<<tempket[i+1]<<" ";
    }
    cout<<" =>\n";
    for (int i = 0; i < ham.norb; i+=2) {
        cout<< tempbra[i]<<tempbra[i+1]<<" ";
    }
    cout<<"\n"<<mat_ele.amplitude<<endl;
}

void printFullMatEle(MatEle& mat_ele, vector<CompactState>& cstatelist)
{
    CompactState tempket = cstatelist[mat_ele.ket], tempbra = cstatelist[mat_ele.bra];
    for (int i = 0; i < ham.norb; i+=2) {
        cout<< tempket[i]<<tempket[i+1]<<" ";
    }
    cout<<" =>\n";
    for (int i = 0; i < ham.norb; i+=2) {
        cout<< tempbra[i]<<tempbra[i+1]<<" ";
    }
    cout<<"\n"<<mat_ele.amplitude<<endl;
}


////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Generate orbital list
////////////////////////////////////////////////////////////////////////////////

vector<Orbital> generate_orblist()
{
    if (ham.norb == 0) {
        cout<< "Orbital Generation Error: norb is zero."<<endl;
    }
    vector<Orbital> orblist(ham.norb);
    auto ite = orblist.begin();
    for (int i = 0; i < ham.mrange; i++)
    {
        Orbital temp1(i, 1);
        *ite = temp1;
        ite++;
    }
    for (int i = 0; i < ham.mrange; i++)
    {
        Orbital temp1(i, 2);
        *ite = temp1;
        ite++;
    }
    return orblist;
}
//m1-m3=k, m1-m4=m
double Vco(int k, int m, double a, double b, int Ns)
{
    int cutoff = 50;
    double v = 0;
    
    for (int q1 = -cutoff; q1 <= cutoff; q1++)
        for (int nm = -cutoff; nm <= cutoff; nm++)
        {
            double q2 = m + nm*Ns;
            double qx = 2 * pi * q1/a;
            double qy = 2 * pi * q2/b;
            double q = sqrt(qx * qx + qy * qy + SmallMomentum);
            if (q != 0)
                v += 1.0/q * exp(-0.5 * q * q) * cos(2 * pi * q1 * k / Ns);
            else
                cout<<"q = 0"<<endl;
        }
    return v/Ns;
}

//m1-m3=k, m1-m4=m
double Vps(int k, int m, double a, double b, int Ns)
{
    int cutoff = 50;
    double v = 0;
    
    for (int q1 = -cutoff; q1 <= cutoff; q1++)
        for (int nm = -cutoff; nm <= cutoff; nm++)
        {
            double q2 = m + nm*Ns;
            double qx = 2 * pi * q1/a;
            double qy = 2 * pi * q2/b;
            double q = sqrt(qx * qx + qy * qy);
            v += 1.0*(1-q*q) * exp(-0.5 * q * q) * cos(2 * pi * q1 * k / Ns);
        }
    return v/Ns;
}

void compute_Coulomb_Forms(vector<Orbital> orblist)
{
    for(int n1 = 0; n1 < ham.mrange; n1++) for(int n2 = 0; n2 < ham.mrange; n2++)
        for(int n3 = 0; n3 < ham.mrange; n3++) for(int n4 = 0; n4 < ham.mrange; n4++)
        {
            if ((n1+n2)%ham.mrange == (n3+n4)%ham.mrange )
            {
                if (ham.interaction == 'c' || ham.interaction == 'C')
                    ham.CoulombForm[n1][n2][n3][n4] = Vco(n1-n3, n1-n4, ham.a, ham.b, ham.mrange);
                else if (ham.interaction == 'p' || ham.interaction == 'P')
                    ham.CoulombForm[n1][n2][n3][n4] = Vps(n1-n3, n1-n4, ham.a, ham.b, ham.mrange);
                else
                {
                    cout<<"Error: interaction form "<<ham.interaction<<" does not exist"<<endl;
                    abort();
                }
                // cout<<n1<<","<<n2<<","<<n3<<","<<n4<<" = "<< ham.CoulombForm[n1][n2][n3][n4]<<endl;
            }
            else
                ham.CoulombForm[n1][n2][n3][n4] = 0;
        }
}

OrbMap generate_orb_idlist(vector<Orbital> orblist)
{
    OrbMap idlist;
    for(int i = 0; i < orblist.size(); i++)
    {
        idlist[orblist[i]] = i;
    }
    return idlist;
}
////////////////////////////////////////////////////////////////////////////////
// Classify pairs according to momentum
////////////////////////////////////////////////////////////////////////////////
PairGroup generate_pair_list(vector<Orbital> orblist, char layer)
{
    PairGroup thelist;
    if (layer == 1)
    {
        for (int id1 = 0; id1 < ham.mrange; id1++)
            for (int id2 = 0; id2 < ham.mrange; id2++)
            {
                if (id1 != id2)
                {
                    thelist[orblist[id1]+orblist[id2]].\
                    push_back(OrbPair(orblist[id1], orblist[id2]));
                }
            }
    }
    else if (layer == 2)
    {
        for (int id1 = ham.mrange; id1 < ham.norb; id1++)
            for (int id2 = ham.mrange; id2 < ham.norb; id2++)
            {
                if (id1 != id2)
                {
                    thelist[orblist[id1]+orblist[id2]].\
                    push_back(OrbPair(orblist[id1], orblist[id2]));
                }
            }
    }
    else
    {
        cout<<"Pair generation error: layer requested out of range"<<endl;
        abort();
    }
    
    return thelist;
}

////////////////////////////////////////////////////////////////////////////////
// Classify states with a given filling according to momentum and spin
////////////////////////////////////////////////////////////////////////////////
int compute_momentum(CompactState &cstate, vector<Orbital>& orblist)
{
    int m = 0;
    for (int i = 0; i < orblist.size(); i++)
    {
        if (cstate[i])
        {
            m += orblist[i].m;
        }
    }
    return (m%ham.mrange);
}

void cs_plusplus(CompactState& cstate)
{
    if (!cstate[ham.norb - 1]) //no carry case
    {
        int i = ham.norb - 1;
        while (!cstate[i]) i--;
        cstate[i] = 0, cstate[i + 1] = 1;
    }
    else           //with carry case
    {
        int i = ham.norb - 1;
        int i2 = 0;
        while (cstate[i])
        {
            i--;
            i2++;
        }
        while (!cstate[i]) i--;
        cstate[i] = 0, cstate[i + 1] = 1;
        i += 2;
        for (int j = i; j < i + i2; j++) cstate[j] = 1;
        if (i + i2 < ham.norb)
            for (int j = i + i2; j < ham.norb; j++) cstate[j] = 0;
    }
}

bool cs_nothighest(CompactState& cstate)
{
    for (int i = ham.norb - 1; i > ham.norb - ham.nele - 1; i--)
    {
        if (!cstate[i]) return true;
    }
    return false;
}

void generate_state_list(vector<Orbital>& orblist, vector<State> &dec_states, ReferenceMap &refMap)
{
    int stateid = StateIdShift;
    CompactState cstate;
    for (int i = 0; i < MaxOrbital; i++) {
        cstate[i] = 0;
    }
    for (int i = 0; i < ham.nele; i++) cstate[i] = 1;
    int m = compute_momentum(cstate, orblist);
    if (m == ham.sector) {
        State temp(cstate);
        temp.state_id = stateid;
        dec_states.push_back(temp);
        refMap[cstate.to_ullong()] = stateid;
        stateid++;
    }
    
    while (cs_nothighest(cstate))
    {
        cs_plusplus(cstate);
        m = compute_momentum(cstate, orblist);
        if (m == ham.sector)
        {
            State temp(cstate);
            temp.state_id = stateid;
            dec_states.push_back(temp);
            refMap[cstate.to_ullong()] = stateid;
            stateid++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Generate matrix elements of interlayer hopping term
////////////////////////////////////////////////////////////////////////////////
int *fast_bra_list;
int *fast_ket_list;
double *fast_amp_list;
long long int fast_size;
long long int fast_count;
void build_hopping_mat(vector<State> &states, ReferenceMap &reference_list, vector<Orbital>& orblist)
{
    MatEle mat_ele;
    Matrix matrix;
    for (auto it : states)
    {
        for (int i = 0; i < ham.mrange; i ++)
        {
            if (it.cstate[i] != it.cstate[i + ham.mrange])
            {
                CompactState temp_cstate(it.cstate);
                int sign_counter = 0;
                if (temp_cstate[i])
                {
                    for (int j = 0; j < i; j++) if (temp_cstate[j]) sign_counter++;
                    temp_cstate[i] = 0;
                    for (int j = 0; j < i + ham.mrange; j++) if (temp_cstate[j]) sign_counter++;
                    temp_cstate[i + ham.mrange] = 1;
                }
                else
                {
                    for (int j = 0; j < i + ham.mrange; j++) if (temp_cstate[j]) sign_counter++;
                    temp_cstate[i + ham.mrange] = 0;
                    for (int j = 0; j < i; j++) if (temp_cstate[j]) sign_counter++;
                    temp_cstate[i] = 1;
                }
                mat_ele.bra = it.state_id-StateIdShift;
                mat_ele.ket = reference_list[temp_cstate.to_ullong()];
                if (mat_ele.ket == 0) {
                    cout << "State reference error: ket state not found while\
                    building kinetic terms."<<endl;
                    abort();
                }
                else
                {
                    mat_ele.ket -= StateIdShift;
                }
                if (mat_ele.bra > mat_ele.ket)
                {
                    if (sign_counter % 2 == 0)
                        mat_ele.amplitude = ham.t;
                    else
                        mat_ele.amplitude = -ham.t;
                    
                    if(abs(mat_ele.amplitude)>SmallDouble)
                    {
                        matrix.push_back(mat_ele);
                    }
                }
            }
        }
    }
    //Then we need to copy the matrix elements from "matrix" to bra, ket and amplitude lists
    fast_size = ham.matrixsize/4 + matrix.size();
    fast_amp_list = new double [fast_size];
    fast_bra_list = new int [fast_size];
    fast_ket_list = new int [fast_size];
    
    fast_count = 0;
    for (auto it : matrix)
    {
        fast_amp_list[fast_count] = it.amplitude;
        fast_bra_list[fast_count] = it.bra;
        fast_ket_list[fast_count] = it.ket;
        fast_count ++;
    }
    
}
//////////////////////////////////////////////////////////////////////////////////////////
// A dry run of Interacting Matrix Elements to count the number of nonzero matrix elements
//////////////////////////////////////////////////////////////////////////////////////////
long long int build_Interaction_mat_dryrun(vector<State> states,
                                 ReferenceMap &reference_list,
                                 PairGroup &pairlist1,
                                 PairGroup &pairlist2,
                                 vector<Orbital> &orblist,
                                 OrbMap &orb_idlist)
{
    int count = 0;
    long long int mat_ele_count = 0;
    for (auto it : states)
    {
        //Then we run through the orbitals on the first layer
        for (int pos1 = 0; pos1 < ham.mrange; pos1++) if (it.cstate[pos1])
            for (int pos2 = 0; pos2 < ham.mrange; pos2++)if (it.cstate[pos2])
                if (pos2 != pos1)
                {
                    Orbital totalprop = orblist[pos1] + orblist[pos2];
                    vector<OrbPair> possible_pairs = pairlist1[totalprop];
                    for (auto it2 : possible_pairs)
                    {
                        int new1 = orb_idlist[it2.orb1];
                        int new2 = orb_idlist[it2.orb2];
                        CompactState tempcstate = it.cstate;
                        tempcstate[pos1] = 0;
                        tempcstate[pos2] = 0;
                        
                        if ((!tempcstate[new1]) && (!tempcstate[new2]))
                        {
                            if(abs(ham.CoulombForm[new1][new2][pos2][pos1])>SmallDouble)
                                mat_ele_count++;
                        }
                    }
                }
        //Then we run through the orbitals on the second layer
        for (int pos1 = ham.mrange; pos1 < ham.norb; pos1++) if (it.cstate[pos1])
            for (int pos2 = ham.mrange; pos2 < ham.norb; pos2++)if (it.cstate[pos2])
                if (pos2 != pos1)
                {
                    Orbital totalprop = orblist[pos1] + orblist[pos2];
                    vector<OrbPair> possible_pairs = pairlist2[totalprop];
                    for (auto it2 : possible_pairs)
                    {
                        int new1 = orb_idlist[it2.orb1];
                        int new2 = orb_idlist[it2.orb2];
                        CompactState tempcstate = it.cstate;
                        tempcstate[pos1] = 0;
                        tempcstate[pos2] = 0;
                        
                        if ((!tempcstate[new1]) && (!tempcstate[new2]))
                        {
                            if(abs(ham.CoulombForm[new1-ham.mrange][new2-ham.mrange][pos2-ham.mrange][pos1-ham.mrange])>SmallDouble)
                                mat_ele_count++;
                        }
                    }
                }
        
        count++;
        if(count %10000 == 0) cout<<"dry run: 10000 states finished, "<<((double)count)/states.size()*100<<"\% finished"<<endl<<mat_ele_count<<" matrix elements in total now"<<endl;
    }
    return mat_ele_count;
}


////////////////////////////////////////////////////////////////////////////////
// Generate Interacting Matrix Elements
////////////////////////////////////////////////////////////////////////////////


void build_Interaction_mat(vector<State> states,
                           ReferenceMap &reference_list,
                           PairGroup &pairlist1,
                           PairGroup &pairlist2,
                           vector<Orbital> &orblist,
                           OrbMap &orb_idlist)
{
    int count = 0;
    for (auto it : states)
    {
        DupMatrix matrix; // the temporary list to store matrix elements
        //Then we run through the orbitals on the first layer
        for (int pos1 = 0; pos1 < ham.mrange; pos1++) if (it.cstate[pos1])
            for (int pos2 = 0; pos2 < ham.mrange; pos2++)if (it.cstate[pos2])
                if (pos2 != pos1)
                {
                    Orbital totalprop = orblist[pos1] + orblist[pos2];
                    vector<OrbPair> possible_pairs = pairlist1[totalprop];
                    for (auto it2 : possible_pairs)
                    {
                        int new1 = orb_idlist[it2.orb1];
                        int new2 = orb_idlist[it2.orb2];
                        CompactState tempcstate = it.cstate;
                        tempcstate[pos1] = 0;
                        tempcstate[pos2] = 0;
                        
                        if ((!tempcstate[new1]) && (!tempcstate[new2]))
                        {
                            if (it2.orb1.layer == it2.orb2.layer)
                            {
                                CompactState tempcstate(it.cstate);
                                
                                int sign_counter = 0;
                                
                                //applying the operators
                                for (int i = 0; i < pos1; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[pos1] = 0;
                                for (int i = 0; i < pos2; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[pos2] = 0;
                                for (int i = 0; i < new2; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[new2] = 1;
                                for (int i = 0; i < new1; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[new1] = 1;
                                
                                int newid = reference_list[tempcstate.to_ullong()];
                                
                                if (newid == 0)
                                {
                                    cout << "State reference Error: ket state not found while building \
                                    interaction matrix term samespin1." <<endl;
                                    
                                    cout<<"from:\t"<<it.cstate<<endl;
                                    cout<<"to:\t"<<tempcstate<<endl<<endl;
                                    abort();
                                }

                                MatEle mat_ele;
                                mat_ele.bra = it.state_id - StateIdShift;
                                mat_ele.ket = newid - StateIdShift;
                                
                                if (mat_ele.bra > mat_ele.ket) //lower triangle, keep the matrix element
                                {
                                    double amplitude = ham.CoulombForm[new1][new2][pos2][pos1];
                                    
                                    if (sign_counter % 2 == 0)
                                        mat_ele.amplitude = amplitude;
                                    else
                                        mat_ele.amplitude = -amplitude;
                                    bra_ket bk(mat_ele.bra, mat_ele.ket);
                                    
                                    if(abs(amplitude)>SmallDouble) matrix[bk] += mat_ele.amplitude;
                                }
                                else if (mat_ele.bra == mat_ele.ket) //diagonal, keep the matrix element/2
                                {
                                    double amplitude = ham.CoulombForm[new1][new2][pos2][pos1];
                                    
                                    if (sign_counter % 2 == 0)
                                        mat_ele.amplitude = amplitude/2;
                                    else
                                        mat_ele.amplitude = -amplitude/2;
                                    bra_ket bk(mat_ele.bra, mat_ele.ket);
                                    
                                    if(abs(amplitude)>SmallDouble) matrix[bk] += mat_ele.amplitude;
                                }
                                //upper triangle, do not keep
                                
                            }
                        }
                    }
                }
        //Then we run through the orbitals on the second layer
        for (int pos1 = ham.mrange; pos1 < ham.norb; pos1++) if (it.cstate[pos1])
            for (int pos2 = ham.mrange; pos2 < ham.norb; pos2++)if (it.cstate[pos2])
                if (pos2 != pos1)
                {
                    Orbital totalprop = orblist[pos1] + orblist[pos2];
                    vector<OrbPair> possible_pairs = pairlist2[totalprop];
                    for (auto it2 : possible_pairs)
                    {
                        int new1 = orb_idlist[it2.orb1];
                        int new2 = orb_idlist[it2.orb2];
                        CompactState tempcstate = it.cstate;
                        tempcstate[pos1] = 0;
                        tempcstate[pos2] = 0;
                        
                        if ((!tempcstate[new1]) && (!tempcstate[new2]))
                        {
                            if (it2.orb1.layer == it2.orb2.layer)
                            {
                                CompactState tempcstate(it.cstate);
                                
                                int sign_counter = 0;
                                
                                //applying the operators
                                for (int i = ham.mrange; i < pos1; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[pos1] = 0;
                                for (int i = ham.mrange; i < pos2; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[pos2] = 0;
                                for (int i = ham.mrange; i < new2; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[new2] = 1;
                                for (int i = ham.mrange; i < new1; i++) if (tempcstate[i]) sign_counter++;
                                tempcstate[new1] = 1;
                                
                                int newid = reference_list[tempcstate.to_ullong()];
                                
                                if (newid == 0)
                                {
                                    cout << "State reference Error: ket state not found while building \
                                    interaction matrix term samespin1." <<endl;
                                    
                                    cout<<"from:\t"<<it.cstate<<endl;
                                    cout<<"to:\t"<<tempcstate<<endl<<endl;
                                    abort();
                                }

                                MatEle mat_ele;
                                mat_ele.bra = it.state_id - StateIdShift;
                                mat_ele.ket = newid - StateIdShift;
                                if (mat_ele.bra > mat_ele.ket) //lower triangle, keep the matrix element
                                {
                                    double amplitude = ham.CoulombForm[new1-ham.mrange][new2-ham.mrange][pos2-ham.mrange][pos1-ham.mrange];
                                    
                                    if (sign_counter % 2 == 0)
                                        mat_ele.amplitude = amplitude;
                                    else
                                        mat_ele.amplitude = -amplitude;
                                    
                                    bra_ket bk(mat_ele.bra, mat_ele.ket);
                                    
                                    if(abs(amplitude)>SmallDouble) matrix[bk] += mat_ele.amplitude;
                                }
                                else if (mat_ele.bra == mat_ele.ket) //diagonal, keep the matrix element/2
                                {
                                    double amplitude = ham.CoulombForm[new1-ham.mrange][new2-ham.mrange][pos2-ham.mrange][pos1-ham.mrange];
                                    
                                    if (sign_counter % 2 == 0)
                                        mat_ele.amplitude = amplitude/2;
                                    else
                                        mat_ele.amplitude = -amplitude/2;
                                    
                                    bra_ket bk(mat_ele.bra, mat_ele.ket);
                                    
                                    if(abs(amplitude)>SmallDouble) matrix[bk] += mat_ele.amplitude;
                                }
                                //upper triangle, do not keep
                                
                            }
                        }
                    }
                }
        
        count++;
        //we need to do something strange with the states here
        for (auto it : matrix)
        {
            fast_amp_list[fast_count] = it.second;
            fast_bra_list[fast_count] = it.first.bra;
            fast_ket_list[fast_count] = it.first.ket;
            fast_count ++;
        }
        
        if(count %10000 == 0) cout<<"10000 states finished, "<<((double)count)/states.size()*100<<"\% finished"<<endl;
    }
}

void matvec(int *size, double *vec_in, double *vec_out, bool *add)
{
    if (!(*add))
        for (int i = 0; i < (*size); i++)
            vec_out[i] = 0.0;
    
    for (long long int i =0; i < fast_size; i++)
    {
        vec_out[fast_ket_list[i]] += fast_amp_list[i] * vec_in[fast_bra_list[i]];
        vec_out[fast_bra_list[i]] += fast_amp_list[i] * vec_in[fast_ket_list[i]];
    }
}

diag_return lanczos_diagonalize(Matrix & matrix, int size, int nevals)
{
    diag_return returnvalue;
    
    vector<double> variance;
    
    lanczos_diag(size, nevals, matvec, returnvalue.eigenvalues, variance);
    
    delete [] fast_bra_list;
    delete [] fast_ket_list;
    delete [] fast_amp_list;
    
    return returnvalue;
}


int run(int norb, int nEle, double a, double t, int sector, int lanczosNE, char interaction)
{
    cout<<"Fractional Quantum Hall System on Torus"<<endl;
    cout<<"Norb: "<<norb<<"\nn_electron: "<<nEle<<"\nt: "<<t<<"\nsector: "<<sector<<endl;
    
    
    ham.nele = nEle;
    ham.lanczosNE=lanczosNE;
    ham.norb = norb;
    ham.mrange = norb/2;
    ham.t=t;
    ham.a = a;
    ham.b = 2*pi * ham.mrange / a;
    ham.interaction = interaction;
    ham.sector = sector;
    
    
    
    vector<Orbital> orblist=generate_orblist();
    for (auto it : orblist) cout<<it<<endl;
    if (ham.norb>MaxOrbital)
    {
        cout<<"Error: Too many orbitals. Raise MaxOrbital"<<endl;
        return 1;
    }
    
    compute_Coulomb_Forms(orblist);
    cout<<"Coulomb factors generated!"<<endl;
    
    //maps from orbital to orbital id //OK
    OrbMap orb_idlist = generate_orb_idlist(orblist);
    cout<<"Orb_idlist generated!"<<endl;
    
    //generate all orbital pairs. (a, b) and (b, a) are considered
    //as different pairs. These two lists are for the two layers
    PairGroup pairlist1 = generate_pair_list(orblist, 1);
    
    for (auto it : pairlist1)
    {
        cout << it.first<<endl;
        for (auto it2 : it.second)
            cout<<it2<<endl;
        cout<<endl;
    }
    PairGroup pairlist2 = generate_pair_list(orblist, 2);
    
    for (auto it : pairlist2)
    {
        cout << it.first<<endl;
        for (auto it2 : it.second)
            cout<<it2<<endl;
        cout<<endl;
    }
    cout<<"================================================"<<endl;
    //generate the list of states with a specific momentum
    //"generation of the LOCAL Hilbert space"
    vector<State> states;
    ReferenceMap reference_list;
    generate_state_list(orblist, states, reference_list);
    
    ////////////////////////////////////////////////////////////////////////////
    //Generate the matrix elements and diagonalize matrices
    //for each momentum sector.
    //This is the most serious job and can be parallelized.
    ////////////////////////////////////////////////////////////////////////////
    
    vector<diag_return> results;
    
    int m = ham.sector;
    cout<<"State Sector: "<<m<<"\t local dimension: "<<states.size()<<endl;
    Matrix matrix;             //the matrix! the most serious thing!
    
    //A dry run first to determine the interaction matrix size
    ham.matrixsize = build_Interaction_mat_dryrun(states, reference_list, pairlist1, pairlist2, orblist, orb_idlist);
    //Then we calculate the hopping matrix first, because it's easier...
    build_hopping_mat(states, reference_list, orblist);
    cout<<"Finished the hopping matrix"<<endl;
    
    build_Interaction_mat(states, reference_list, pairlist1, pairlist2, orblist, orb_idlist);
    cout<<"Finished the interaction matrix"<<endl;
    
    cout<<"Using Haldane's Lanczos"<<endl;
    diag_return diag_result=lanczos_diagonalize(matrix,states.size(), ham.lanczosNE);
    diag_result.sector_indicator=m;
    results.push_back(diag_result);
    
    cout<<"Here is the diagonalization result: "<<endl;
    ofstream printresult;
    printresult.open("printresult.txt");
    int evcount = ham.lanczosNE;
    for(auto it : results)
    {
        int evcount = ham.lanczosNE;
        bool short_output_mode = (evcount > 0);
        for (auto& it2 : it.eigenvalues)
        {
            if (short_output_mode) {
                evcount--;
                if (evcount < 0)
                    break;
            }
            cout<<it.sector_indicator<<" ";
            if(abs(it2)<SmallDouble) cout<<" E = 0"<<endl;
            else cout<<" E = "<<it2<<endl;
        }
        for (auto& it2 : it.eigenvalues)
        {
            printresult<<it.sector_indicator<<" ";
            if(abs(it2)<SmallDouble) printresult<<" E = 0"<<endl;
            else printresult<<" E = "<<it2<<endl;
        }
        
    }
    return 0;
}

int main()
{
    int norb, nele, nEv;
    double a, t;
    char interaction;
    int sector;
    cout <<"norb, nele, a, t, m_sector, nEv, interaction(c for Coulomb, p for pseudopotential)"<<endl;
    cin>>norb>>nele>>a>>t>>sector>>nEv>>interaction;
    run(norb, nele, a, t, sector,nEv, interaction);
}
