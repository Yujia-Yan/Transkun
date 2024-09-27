import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .Util import checkpointByPass,checkpointSequentialByPass


class RMSNorm(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(dim = -1, keepdim=True)
        return x* torch.rsqrt(var+ self.eps)


class TiedDropout(nn.Module):
    def __init__(self, dropoutProb, axis):
        super().__init__()
        self.dropout = nn.Dropout(dropoutProb)
        self.axis = axis
    
    def forward(self, x):

        if self.training:
            dropShape = list(x.shape)
            dropShape[self.axis] = 1
            mask = torch.ones(*dropShape, device = x.device)
        
            return self.dropout(mask)*x
        else:
            return x


class LearnableSpatialPositionEmbedding(nn.Module):
    def __init__(self, embedSize, coordDim, gamma = 10.0, dropoutProb = 0.0):
        super().__init__()

        self.gamma = gamma
        self.proj = nn.Linear(coordDim, embedSize)

        self.mlp = nn.Sequential(
                nn.Linear(embedSize, 4*embedSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(4*embedSize, embedSize))


        self.dropout = nn.Dropout(dropoutProb)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.normal_(self.proj.weight, std = 1/self.gamma)
        nn.init.uniform_(self.proj.bias, a = -math.pi, b = math.pi)

    """
    arguments:
        indices [nBatch, nDimCoordinates]
    """
    def forward(self, *coords):
        device = self.proj.weight.device
        coords = torch.meshgrid(coords, indexing="ij")
        coord = torch.stack(coords, dim = -1)

        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z

    def forwardWithCoordVec(self, coord):
        device = self.proj.weight.device

        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z

class ResBlock(nn.Module):
    def __init__(self, module, size, prenorm = True, dropoutProb =0.0):
        super().__init__()
        self.module = module
        self.norm = RMSNorm()

        # LayerScale
        self.scale = nn.Parameter(torch.ones(size)*1e-2)
        self.dropout = nn.Dropout(dropoutProb)

    def forward(self, x, *args):
        return x + self.dropout(self.module(self.norm(x), *args))*self.scale

class SelfAttnWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        shape = x.shape
        x = x.flatten(0,1)
        result, _ = self.module(x,x,x)

        result = result.unflatten(0, shape[:2])
        return result


"""
The customized MHA layer using the approximated attention
adapted from the pytorch implementation
"""
class MultiHeadAttentionKernel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0., k_dim = None, v_dim = None, fourierSize = 32, kernel = "fourier", hiddenFactor = 1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel = kernel


        hiddenSize = math.ceil(hiddenFactor*embed_dim)

        # make sure hiddenSize to be divisible by num_heads
        self.head_dim = int(math.ceil(hiddenSize/num_heads))
        hiddenSize = self.head_dim*num_heads



        if k_dim is None:
            k_dim = embed_dim

        if v_dim is None:
            v_dim = embed_dim


        self.fourierSize = fourierSize


        self.q_proj_weight = nn.Parameter(torch.empty(( embed_dim, hiddenSize)))
        self.k_proj_weight = nn.Parameter(torch.empty(( k_dim, hiddenSize)))
        self.v_proj_weight = nn.Parameter(torch.empty(( v_dim, hiddenSize)))
        self.out_proj = nn.Linear(hiddenSize, embed_dim)

        if kernel is not None:
            self.gamma = nn.Parameter(torch.tensor(1.0))
            self.norm = RMSNorm()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

    def forward(self, query, key = None, value = None):
        if key == None:
            key = query

        if value == None:
            value = key

        q = query@self.q_proj_weight
        k = key@self.k_proj_weight
        v = value@self.v_proj_weight

        # split into heads
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(-2,-3)
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.transpose(-2,-3)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.transpose(-2,-3)

        if self.kernel is not None:
            raise NotImplementedError
        else:
            fetched = F.scaled_dot_product_attention(q,k,v)

        fetched = fetched.transpose(-2,-3).flatten(-2,-1)

        result = self.out_proj(fetched)

        return result


class BasicBlock(nn.Module):
    def __init__(self,
            inputSize,
            num_heads,
            fourierSize,
            hiddenFactor = 2,
            hiddenFactorAttn = 1,
            approxKernels = [None,
                None,
                None, None],
            enabled = ["F", "T", "All0", "0All"],
            dropoutProb = 0.0):
        super().__init__()

        fnnHiddenSize = int(math.ceil(inputSize*hiddenFactor))

        self.enabled = enabled

        if "F" in enabled:
            self.mhaBlockF = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = fourierSize,
                        kernel = approxKernels[0],
                        hiddenFactor = hiddenFactorAttn),

                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

            self.fnnBlockF  = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

        if "T" in enabled:
            self.mhaBlockT = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = fourierSize,
                        kernel = approxKernels[1],
                        hiddenFactor = hiddenFactorAttn
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )
            self.fnnBlockT  = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )
        if "All0" in enabled or "0All" in enabled :
            self.mhaBlockAll0 = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = fourierSize,
                        kernel = approxKernels[2],
                        hiddenFactor = hiddenFactorAttn
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )
            self.fnnBlockAll0  = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

        if "FT" in enabled:
            self.mhaBlockFT = ResBlock(
                    MultiHeadAttentionKernel(
                        inputSize,
                        num_heads=num_heads,
                        fourierSize = 64,
                        kernel = "positive",
                        hiddenFactor = hiddenFactorAttn
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )

            self.fnnBlockFT = ResBlock(
                    nn.Sequential(
                        nn.Linear(inputSize, fnnHiddenSize),
                        nn.GELU(),
                        nn.Dropout(dropoutProb),
                        nn.Linear(fnnHiddenSize, inputSize),
                        ),
                    size = inputSize,
                    dropoutProb = dropoutProb
                    )



    def forward(self, x, mem = None, crossAttn = False):
        # x: [N, T, F, D]
        nT = x.shape[-3]
        nF = x.shape[-2]

        inShape = x.shape

        crossAttn = True 

        if mem is None:
            mem = x
            crossAttn = False

        h = x

        if "F" in self.enabled:
            # print("F")
            h = self.mhaBlockF(h, mem)
            h = self.fnnBlockF(h)

        # change to [N, F, T, D]
        h = h.transpose(-3, -2)
        mem = mem.transpose(-3, -2)

        if "T" in self.enabled:
            # print("T")
            # all attends to the aggregated track
            # if crossAttn:
                # h = self.mhaBlockT(h, mem[..., 0:1, :, :])
            # else:
            h = self.mhaBlockT(h, mem)
            h = self.fnnBlockT(h)

        if "All0" in self.enabled or "0All" in self.enabled:
            h0, h1 = h.split([1, h.shape[-3]-1], dim = -3)

            if "All0" in self.enabled:
                h1 = self.mhaBlockAll0(h1, mem[..., 0:1, :,:])

            if "0All" in self.enabled:
                # h0 = h[..., 0, :, :]
                h0 = self.mhaBlockAll0(h0, mem.flatten(-3,-2).unsqueeze(-3))
                # print(h.shape)

            h = torch.cat([h0, h1], dim = -3)
            h = self.fnnBlockAll0(h)
            



        # change to [N, F*T, D]
        if "FT" in self.enabled:
            h = h.flatten(-3,-2)
            mem = mem.flatten(-3, -2)

            h = self.mhaBlockFT(h, mem)
            h = self.fnnBlockFT(h)
            h = h.unflatten(-2, (nF, nT))

        # change back to the orignal shape
        h = h.transpose(-3, -2)

        outShape = h.shape
        assert inShape == outShape


        return h





"""
What if we even simplify the scoring module?
"""
class ScaledInnerProductIntervalScorer(nn.Module):
    def __init__(self, size, expansionFactor = 1, dropoutProb = 0.0, withScoreEps = False, lengthScaling="linear"):
        super().__init__()

        self.size = size

        if not withScoreEps:
            self.map = nn.Sequential(
                    nn.Linear(size, 2*size*expansionFactor+1), # only inner product plus diagonal
                    )
        else:
            self.map = nn.Sequential(
                    nn.Linear(size, 2*size*expansionFactor+1 + 1), 
                    )

        
        self.dropout = nn.Dropout(dropoutProb)

        self.expansionFactor = expansionFactor

        self.lengthScaling = lengthScaling

    def forward(self, ctx):


        q, k, diag = (self.map(ctx)).split([self.size*self.expansionFactor,
            self.size*self.expansionFactor, 1], dim = -1)

        # print(q.std(), k.std(), b.std())
        q = q/math.sqrt(q.shape[-1])

        # part1 innerproduct
        S = torch.einsum("iped, ipbd-> ipeb", q, k)
        # diagS = S.diagonal(dim1= -2, dim2=-1)

        tmpIdx_e = torch.arange(S.shape[-2], device = S.device)
        tmpIdx_b = torch.arange(S.shape[-1], device = S.device)
        len_eb = (tmpIdx_e.unsqueeze(-1)- tmpIdx_b.unsqueeze(0)).abs()

        if self.lengthScaling == "linear":
            S = S*(len_eb)
        elif self.lengthScaling == "sqrt":
            S = S*(len_eb).float().sqrt()
        elif self.lengthScaling == "none":
            pass
        else:
            raise Exception("Unrecognized lengthScaling")



        diagM= torch.diag_embed(diag.squeeze(-1))

        S = S + diagM

        # dummy eps score for testing
        b = diag*0.0
        b = b[...,1:, 0]

        S = S.permute(2,3, 0, 1).contiguous()
        b = b.permute(2,0,1).contiguous()
        return S, b


class Backbone(nn.Module):
    def __init__(self,
            inputSize,
            baseSize,
            posEmbedInitGamma,
            nHead,
            fourierSize = 16,
            hiddenFactor = 2,
            hiddenFactorAttn = 1,
            expansionFactor = 1,
            dropoutProb = 0.0,
            nLayers= 4,
            enabledAttn = ["F", "T"],
            useGradientCheckpoint = True,
            downsampleF = True,
            upsampleProjOnly = True,
            ) :
        super().__init__()

        self.posEmbedBuilder = LearnableSpatialPositionEmbedding(
                baseSize,
                coordDim = 1,
                gamma = posEmbedInitGamma,
                dropoutProb= dropoutProb)




        self.inputConv = nn.Conv2d(inputSize, baseSize, kernel_size = 3, padding =1)
        self.dropoutTied = TiedDropout(dropoutProb, axis = -3)

        # temporal patch of size 8

        if not downsampleF:
            # path size 8x1
            self.downConv = nn.Sequential(
                    nn.ConstantPad2d( (0,0, 4, 3), value = 0.0),
                    nn.Conv2d(baseSize, baseSize*2, 3, padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*2),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*2, baseSize*4, 3,  padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),
                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1),
                    nn.GroupNorm(4, baseSize*4),
                    )
        else:
            # this time the patch size is 8x 4
            self.downConv = nn.Sequential(
                    nn.ConstantPad2d( (2, 1, 4, 3), value = 0.0),
                    nn.Conv2d(baseSize, baseSize*2, 3, padding = 1, stride = (2,1)),
                    nn.GroupNorm(4, baseSize*2),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*2, baseSize*4, 3,  padding = 1, stride = (2,2)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),

                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1, stride = (2,2)),
                    nn.GroupNorm(4, baseSize*4),
                    nn.GELU(),
                    nn.Dropout2d(dropoutProb),
                    nn.Conv2d(baseSize*4, baseSize*4, 3,  padding = 1),
                    nn.GroupNorm(4, baseSize*4),
                    )


        self.upConv1dSkip =  nn.ConvTranspose1d(baseSize*4, baseSize*expansionFactor, 8, stride = 8)
        if not upsampleProjOnly:
            self.upConv1d = nn.Sequential(
                    nn.ConvTranspose1d(baseSize*4, baseSize*4, 2, stride = 2),
                    nn.Conv1d(baseSize*4, baseSize*4, 3, padding = 1),
                    nn.GroupNorm(4,  baseSize*4),
                    nn.GELU(),
                    nn.ConvTranspose1d(baseSize*4, baseSize*2, 2, stride = 2),
                    nn.Conv1d(baseSize*2, baseSize*2, 3, padding = 1),
                    nn.GroupNorm(4,  baseSize*2),
                    nn.GELU(),
                    nn.ConvTranspose1d(baseSize*2, baseSize, 2, stride = 2),
                    nn.Conv1d(baseSize, baseSize, 3, padding = 1),
                    )
        self.upsampleProjOnly = upsampleProjOnly

        self.posEmbedBuilderAttnTF = LearnableSpatialPositionEmbedding(
                baseSize*4,
                coordDim = 2,
                gamma = posEmbedInitGamma,
                dropoutProb=dropoutProb)


        self.posEmbedBuilderAttnTE = LearnableSpatialPositionEmbedding(
                baseSize*4,
                coordDim = 2,
                gamma = posEmbedInitGamma, dropoutProb = dropoutProb)


        
        encoderLayers = [BasicBlock( inputSize = baseSize*4,
            num_heads = nHead,
            fourierSize = fourierSize,
            dropoutProb = dropoutProb,
            hiddenFactor = hiddenFactor,
            hiddenFactorAttn = hiddenFactorAttn,
            enabled = enabledAttn
            ) for i in range(nLayers)]

        self.encoderLayers = nn.ModuleList(encoderLayers)

        self.normEncoder = nn.Identity()

        self.useGradientCheckpoint = useGradientCheckpoint

        self.dropout = nn.Dropout(dropoutProb)



    def forward(self, x, outputIndices):
        if self.useGradientCheckpoint or self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass
        # x: [N, T, F, D]

        # change to
        # x: [N, D, T, F]
        x = x.permute(0, 3,1,2)


        nT = x.shape[-2]
        nF = x.shape[-1]

        # append the first position embedding

        coord_F = torch.arange(x.shape[-1], device = x.device).float()

        posEmbedInputConv = self.posEmbedBuilder(coord_F)
        posEmbedInputConv = posEmbedInputConv.transpose(-1,-2).unsqueeze(-2)


        # downsample along the the time

        # downsample 4 times for reducing the computation cost
        h = self.inputConv(x)+posEmbedInputConv
        h = self.downConv(h)


        # change to [N, T, F, D] shape
        h = h.permute(0, 2, 3, 1)

        # append 1 time and 1 frequency aggregation track
        h = F.pad(h, (0,0,1,0, 1, 0))

        ################ transformer encoders
        coord_F = torch.arange(h.shape[-2], device = x.device).float()

        coord_T = torch.arange(h.shape[-3], device = x.device).float()
        outputIndices =  outputIndices.float()


        posEmbed = self.posEmbedBuilderAttnTF(coord_T, coord_F)

        posEmbedTgt = self.posEmbedBuilderAttnTE(coord_T,  outputIndices)

        posEmbedTgt = posEmbedTgt.unsqueeze(0).repeat(h.shape[0], 1,1,1)

        h = h + posEmbed
        hTarget = posEmbedTgt



        hAll = torch.cat( [h, hTarget], dim = -2)


        for l in self.encoderLayers:
            hAll = checkpoint(l, hAll)


        h, hTarget = hAll.split([h.shape[-2], hTarget.shape[-2]], dim = -2)

        # print(hTarget.std(), hTarget.mean())

        # remove the t=0 pooling track
        hTarget = hTarget[..., 1:, :, :]

        # do 1d upsampling
        hTarget = hTarget.permute(0, 2, 3,1)

        # now h:[N, P, D,T ]

        hTarget = hTarget.flatten(0,1)
        # now h:[N*P, D,T ]
        if not self.upsampleProjOnly:
            hTarget = self.upConv1d(hTarget)+ self.upConv1dSkip(hTarget)
        else:
            # it seems that this linear projection is good enough
            hTarget = self.upConv1dSkip(hTarget)

        hTarget = hTarget.unflatten(0, (x.shape[0], len(outputIndices)))

        hTarget = hTarget[..., :nT]

        hTarget = hTarget.permute(0, 1,3,2)

        # the outputShape: [N, P, T, D]


        return hTarget

if __name__ == "__main__":


    # v = torch.arange(1, 22).unflatten(-1, (-1,3))

    # createBandedTriangularMatrix(v)
    
    device = "cuda"
    ctx = torch.randn(4, 4*90, 201, 32).to(device)
    scorer = IntervalScorerIPOD_V2(32, nOffDiagonals = 2).to(device)
    scorer(ctx)

    exit()
    # from .Layers_ablation import PairwiseFeatureBatch

    # ctx = torch.randn(500, 4*90 ,32).to(device)

    # scorer = PairwiseFeatureBatch(inputSize = 32, outputSize = 1).to(device)

    # t1 = time.time()

    # for i in range(100):
        # S, S_skip = scorer(ctx, nBlock= 3200)

    # t2 = time.time()
    # print(S.shape, S_skip.shape)
    # print(t2-t1)

    

    # exit()

    # x = torch.randn(1,1,1000, 256)/10
    # # x = F.normalize(x, dim = -1)
    # ff = RandomFourierFeatureMap(1,1, 256, 128, gamma =1, kernel ="innerProduct")

    # z = ff(x)
    # # print(z.shape)
    # # c = math.sqrt(1/z.shape[-1]/2)
    # A = (z)@(z.transpose(-1,-2))
    # # print(A.shape)
    # # print(z)
    # print(z.sum(-1)/math.sqrt(z.shape[-1]))
    # exit()


    # A2 = (x@x.transpose(-1,-2)).exp()
    # # A2 = (x@x.t()/256).exp()

    # # A2 = (-((x.unsqueeze(-2) - x.unsqueeze(-3))**2).sum(-1)/2*(ff.gamma**2)).exp()
    # # print(A2.shape)
    # print((A2-A).abs().mean())
    # print(A)
    # print(A2)
    # print(A-A2)





    # A2 = (-((x.unsqueeze(-2) @ x.unsqueeze(-3)))).exp()


    # then we try to approximate multihead attention
    torch.manual_seed(42)

    k = torch.randn(1, 1, 100, 256)/4
    q = torch.randn(1, 1, 100, 256)/4
    v = torch.randn(1, 1, 100, 256)

    # z_k2 = ff2(k)
    # z_q2 = ff2(q)

    # print(z_k.shape)
    # print(z_q.shape)

    # resultAll =[]
    # resultAllDebiased = []
    # for i in range(1000):
        # ff = RandomFourierFeatureMap(1,1, 256, 32, gamma =1/math.sqrt((math.sqrt(256))), kernel ="innerProduct")
        # z_k = ff(k)
        # z_q = ff(q)

        
        # tmp = z_k.transpose(-1, -2).sum(dim = -1, keepdim=True)
        # denom = z_q@tmp

        # fetched = z_q@(z_k.transpose(-1,-2)@v)/denom

        # tk = (z_k.transpose(-1,-2) @v)

        # # print((z_q@tk).shape)
        # # print(z_q.shape)
        # # print(z_q.shape,tk.shape)
        # t1_numerator = z_q@tk
        # t2_numerator = (torch.einsum( "bhqd, bhdv -> bhqvd", z_q, tk))
        # numerator = (t1_numerator).unsqueeze(-1)- t2_numerator
        # # print(numerator.shape)

        # # print(numerator.mean(dim = -1))
        # # print(t1_numerator)

        # # exit()


        # t1_denom= torch.einsum("bhqd, bhkd -> bhq", z_q, z_k)
        # t2_denom= torch.einsum("bhqd, bhkd -> bhqd", z_q, z_k)
        # denom = t1_denom.unsqueeze(-1)- t2_denom

        # # print(t1_denom - denom.mean(dim = -1))



        # fetchedJack = (numerator/ denom.unsqueeze(-2)).mean(dim = -1)
        # # print(fetchedJack.shape)

        # N = z_k.shape[-1]

        # fetchedDebiased = fetched*N- (N-1)*fetchedJack
        # # print((fetched-fetchedJack).abs().max())
        # # exit()




        # # exit()


        # # debiasing with jackknife

        # resultAll.append(fetched)
        # resultAllDebiased.append(fetchedDebiased)

    # fetched = torch.stack(resultAll, dim = -1).mean(dim = -1)     
    # fetchedDebiased = torch.stack(resultAllDebiased, dim = -1).mean(dim = -1)     
    # # debiasing with jacknife
    # print(tmp.shape)



    # fetched2 = F.scaled_dot_product_attention(q, k ,v)
    # print((fetched2-fetched).abs().mean())
    # print("debias", (fetched2-fetchedDebiased).abs().mean())

    # import time
    # mha = MultiHeadAttentionKernel(256, 2, fourierSize=64)
    # mha = mha.cuda()

    # q = q.cuda()
    # k = k.cuda()
    # v = v.cuda()

    # t1 = time.time()
    # for i in range(1000):
        # fetched = mha(q,k,v)
    # t2 = time.time()
    # print(t2-t1)
    # print(fetched.shape)

    # testing the transkun backend

    # input shape
    def computeParamSize(module):
        total_params = sum(p.numel() for p in module.parameters())

        # Convert to millions
        total_params_millions = total_params / 1e6
        return total_params_millions

    x = torch.randn(4,  432*2-5, 229, 6).cuda()

    # the following two should be consistent, trained model can be converted
    # preLayerCompare = BackboneFinal(inputSize = 6,
            # baseSize = 64,
            # posEmbedInitGamma= 10.0,
            # nHead = 4,
            # nLayers = 6,
            # hiddenFactor = 1.,
            # hiddenFactorAttn = 0.5,
            # # fourierSize = 32,
            # expansionFactor = 1,
            # downsampleF = True).cuda()

    # preLayer = BackboneFinalV2(inputSize = 6,
            # inputProjSize = 64,
            # embedSize = 256,
            # patchShape= (8,4),
            # posEmbedInitGamma= 1.0,
            # nHead = 4,
            # nLayers = 6,
            # hiddenFactor = 1.,
            # expansionFactor = 1/4, # for dimension of interval scoring
            # hiddenFactorAttn = 0.5,
            # ).cuda()

    # print("#Param(M):", computeParamSize(preLayerCompare))
    # print("#Param(M):", computeParamSize(preLayer))
    # add position Embeeding
    # indices for the spatial positions
    # posEmbedBuilder = LearnableSpatialPositionEmbedding2D(64, gamma = 10.0)
    # posEmbed1 = posEmbedBuilder(size1 = x.shape[-2], size2 = x.shape[-3])

    # print(posEmbed1.shape)
    preLayer = BackboneFinalV2(inputSize = 6,
            inputProjSize = 64,
            embedSize = 256,
            embedSizeOut = 200,
            useLastKLayers = 1,
            patchShape= (16,8),
            posEmbedInitGamma= 1.0,
            nHead = 8,
            nLayers = 6,
            hiddenFactor = 2.,
            hiddenFactorAttn = 1,
            ).cuda()


    print("#Param(M):", computeParamSize(preLayer))


    for i in range(1000):
        t1 = time.time()

        preLayer.zero_grad()
        print(i)
        print(x.shape)
        z = preLayer(x, outputIndices = torch.arange(90, device = device))
        print(z.shape)
        z.sum().backward()

        t2 = time.time()
        print(t2-t1)

