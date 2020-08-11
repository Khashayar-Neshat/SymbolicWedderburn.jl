#= GAP code:
H := [];
for i in [2..30] do
    Add(H, List(AllSmallGroups(i), G->Image(IsomorphismPermGroup(G))));
od;
PrintTo("/tmp/groups.gap", H);
=#

#=julia code
GAPgroups_str = join(readlines("/tmp/group.gap"), "");
GAPgroups_str = replace(GAPgroups_str, "Group"=>"\nPermGroup");
GAPgroups_str = replace(GAPgroups_str, r" *"=>"");
perm_regex = r"((\(\d+(,\d+)*\)?)+)";
print(Meta.parse(replace(GAPgroups_str, perm_regex=> s"perm\"\1\"")))
=#

const SmallPermGroups = Dict(
    i + 1 => x
    for
    (i, x) in enumerate([
        [PermGroup([perm"(1,2)"])],
        [PermGroup([perm"(1,2,3)"])],
        [PermGroup([perm"(1,2,3,4)"]), PermGroup([perm"(1,2)", perm"(3,4)"])],
        [PermGroup([perm"(1,2,3,4,5)"])],
        [
            PermGroup([perm"(1,2)(3,6)(4,5)", perm"(1,3,5)(2,4,6)"]),
            PermGroup([perm"(1,2)", perm"(3,4,5)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7)"])],
        [
            PermGroup([perm"(1,2,3,4,5,6,7,8)"]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6)"]),
            PermGroup([
                perm"(1,2)(3,8)(4,6)(5,7)",
                perm"(1,3)(2,5)(4,7)(6,8)",
                perm"(1,4)(2,6)(3,7)(5,8)",
            ]),
            PermGroup([
                perm"(1,2,4,6)(3,8,7,5)",
                perm"(1,3,4,7)(2,5,6,8)",
                perm"(1,4)(2,6)(3,7)(5,8)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6)"]),
        ],
        [
            PermGroup([perm"(1,2,3,4,5,6,7,8,9)"]),
            PermGroup([perm"(1,2,3)", perm"(4,5,6)"]),
        ],
        [
            PermGroup([
                perm"(1,2)(3,10)(4,9)(5,8)(6,7)",
                perm"(1,3,5,7,9)(2,4,6,8,10)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6,7)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11)"])],
        [
            PermGroup([
                perm"(1,2,3,5)(4,10,7,12)(6,11,9,8)",
                perm"(1,3)(2,5)(4,7)(6,9)(8,11)(10,12)",
                perm"(1,4,8)(2,6,10)(3,7,11)(5,9,12)",
            ]),
            PermGroup([perm"(1,2,3)", perm"(4,5,6,7)"]),
            PermGroup([
                perm"(1,2,5)(3,7,12)(4,11,9)(6,10,8)",
                perm"(1,3)(2,6)(4,8)(5,9)(7,11)(10,12)",
                perm"(1,4)(2,7)(3,8)(5,10)(6,11)(9,12)",
            ]),
            PermGroup([
                perm"(1,2)(3,5)(4,10)(6,8)(7,12)(9,11)",
                perm"(1,3)(2,5)(4,7)(6,9)(8,11)(10,12)",
                perm"(1,4,8)(2,6,10)(3,7,11)(5,9,12)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6,7)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13)"])],
        [
            PermGroup([
                perm"(1,2)(3,14)(4,13)(5,12)(6,11)(7,10)(8,9)",
                perm"(1,3,5,7,9,11,13)(2,4,6,8,10,12,14)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6,7,8,9)"]),
        ],
        [PermGroup([perm"(1,2,3)", perm"(4,5,6,7,8)"])],
        [
            PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)"]),
            PermGroup([perm"(1,2,3,4)", perm"(5,6,7,8)"]),
            PermGroup([
                perm"(1,2,5,8)(3,12,10,16)(4,7,11,14)(6,15,13,9)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,12)(8,13)(11,15)(14,16)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,12)(8,14)(10,15)(13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([
                perm"(1,2,5,8)(3,12,10,16)(4,7,11,14)(6,15,13,9)",
                perm"(1,3,4,9)(2,6,7,12)(5,10,11,15)(8,13,14,16)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,12)(8,14)(10,15)(13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6,7,8,9,10)"]),
            PermGroup([
                perm"(1,2,4,7,5,8,11,14)(3,13,9,16,10,6,15,12)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,12)(8,13)(11,15)(14,16)",
                perm"(1,4,5,11)(2,7,8,14)(3,9,10,15)(6,12,13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([
                perm"(1,2)(3,12)(4,14)(5,8)(6,9)(7,11)(10,16)(13,15)",
                perm"(1,3)(2,6)(4,15)(5,10)(7,16)(8,13)(9,11)(12,14)",
                perm"(1,4,5,11)(2,7,8,14)(3,9,10,15)(6,12,13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([
                perm"(1,2,5,8)(3,12,10,16)(4,14,11,7)(6,15,13,9)",
                perm"(1,3)(2,6)(4,15)(5,10)(7,16)(8,13)(9,11)(12,14)",
                perm"(1,4,5,11)(2,7,8,14)(3,9,10,15)(6,12,13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([
                perm"(1,2,5,8)(3,12,10,16)(4,14,11,7)(6,15,13,9)",
                perm"(1,3,5,10)(2,6,8,13)(4,15,11,9)(7,16,14,12)",
                perm"(1,4,5,11)(2,7,8,14)(3,9,10,15)(6,12,13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6,7,8)"]),
            PermGroup([
                perm"(1,2)(3,13)(4,7)(5,8)(6,10)(9,16)(11,14)(12,15)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,12)(8,13)(11,15)(14,16)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,12)(8,14)(10,15)(13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([
                perm"(1,2,5,8)(3,13,10,6)(4,7,11,14)(9,16,15,12)",
                perm"(1,3,5,10)(2,6,8,13)(4,9,11,15)(7,12,14,16)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,12)(8,14)(10,15)(13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([
                perm"(1,2)(3,13)(4,7)(5,8)(6,10)(9,16)(11,14)(12,15)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,12)(8,13)(11,15)(14,16)",
                perm"(1,4,5,11)(2,7,8,14)(3,9,10,15)(6,12,13,16)",
                perm"(1,5)(2,8)(3,10)(4,11)(6,13)(7,14)(9,15)(12,16)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6)", perm"(7,8)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)"])],
        [
            PermGroup([
                perm"(1,2)(3,18)(4,12)(5,17)(6,9)(7,16)(8,15)(10,14)(11,13)",
                perm"(1,3,7,4,8,13,9,14,17)(2,5,10,6,11,15,12,16,18)",
                perm"(1,4,9)(2,6,12)(3,8,14)(5,11,16)(7,13,17)(10,15,18)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6,7,8,9,10,11)"]),
            PermGroup([
                perm"(1,2)(3,5)(4,12)(6,9)(7,10)(8,16)(11,14)(13,18)(15,17)",
                perm"(1,3,7)(2,5,10)(4,8,13)(6,11,15)(9,14,17)(12,16,18)",
                perm"(1,4,9)(2,6,12)(3,8,14)(5,11,16)(7,13,17)(10,15,18)",
            ]),
            PermGroup([
                perm"(1,2)(3,10)(4,12)(5,7)(6,9)(8,18)(11,17)(13,16)(14,15)",
                perm"(1,3,7)(2,5,10)(4,8,13)(6,11,15)(9,14,17)(12,16,18)",
                perm"(1,4,9)(2,6,12)(3,8,14)(5,11,16)(7,13,17)(10,15,18)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5)", perm"(6,7,8)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)"])],
        [
            PermGroup([
                perm"(1,2,3,5)(4,18,7,20)(6,19,9,16)(8,14,11,17)(10,15,13,12)",
                perm"(1,3)(2,5)(4,7)(6,9)(8,11)(10,13)(12,15)(14,17)(16,19)(18,20)",
                perm"(1,4,8,12,16)(2,6,10,14,18)(3,7,11,15,19)(5,9,13,17,20)",
            ]),
            PermGroup([perm"(1,2,3,4)", perm"(5,6,7,8,9)"]),
            PermGroup([
                perm"(1,2,3,5)(4,10,19,17)(6,11,20,12)(7,13,16,14)(8,18,15,9)",
                perm"(1,3)(2,5)(4,19)(6,20)(7,16)(8,15)(9,18)(10,17)(11,12)(13,14)",
                perm"(1,4,8,12,16)(2,6,10,14,18)(3,7,11,15,19)(5,9,13,17,20)",
            ]),
            PermGroup([
                perm"(1,2)(3,5)(4,18)(6,16)(7,20)(8,14)(9,19)(10,12)(11,17)(13,15)",
                perm"(1,3)(2,5)(4,7)(6,9)(8,11)(10,13)(12,15)(14,17)(16,19)(18,20)",
                perm"(1,4,8,12,16)(2,6,10,14,18)(3,7,11,15,19)(5,9,13,17,20)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6,7,8,9)"]),
        ],
        [
            PermGroup([
                perm"(1,2,4)(3,8,16)(5,10,12)(6,14,7)(9,20,19)(11,21,15)(13,18,17)",
                perm"(1,3,6,9,12,15,18)(2,5,8,11,14,17,20)(4,7,10,13,16,19,21)",
            ]),
            PermGroup([perm"(1,2,3)", perm"(4,5,6,7,8,9,10)"]),
        ],
        [
            PermGroup([
                perm"(1,2)(3,22)(4,21)(5,20)(6,19)(7,18)(8,17)(9,16)(10,15)(11,14)(12,13)",
                perm"(1,3,5,7,9,11,13,15,17,19,21)(2,4,6,8,10,12,14,16,18,20,22)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6,7,8,9,10,11,12,13)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)"])],
        [
            PermGroup([
                perm"(1,2,3,6,4,7,9,13)(5,16,10,21,11,22,17,24)(8,18,14,19,15,23,20,12)",
                perm"(1,3,4,9)(2,6,7,13)(5,10,11,17)(8,14,15,20)(12,18,19,23)(16,21,22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([perm"(1,2,3)", perm"(4,5,6,7,8,9,10,11)"]),
            PermGroup([
                perm"(1,2,6)(3,8,20)(4,16,13)(5,9,15)(7,14,10)(11,18,24)(12,23,21)(17,22,19)",
                perm"(1,3,5,11)(2,7,9,17)(4,19,12,10)(6,13,15,21)(8,23,18,16)(14,24,22,20)",
                perm"(1,4,5,12)(2,8,9,18)(3,10,11,19)(6,14,15,22)(7,16,17,23)(13,20,21,24)",
                perm"(1,5)(2,9)(3,11)(4,12)(6,15)(7,17)(8,18)(10,19)(13,21)(14,22)(16,23)(20,24)",
            ]),
            PermGroup([
                perm"(1,2,4,7)(3,13,9,6)(5,16,11,22)(8,19,15,12)(10,24,17,21)(14,18,20,23)",
                perm"(1,3,4,9)(2,6,7,13)(5,10,11,17)(8,14,15,20)(12,18,19,23)(16,21,22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([
                perm"(1,2)(3,6)(4,7)(5,16)(8,12)(9,13)(10,21)(11,22)(14,18)(15,19)(17,24)(20,23)",
                perm"(1,3,4,9)(2,6,7,13)(5,10,11,17)(8,14,15,20)(12,18,19,23)(16,21,22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([
                perm"(1,2)(3,13)(4,7)(5,16)(6,9)(8,12)(10,24)(11,22)(14,23)(15,19)(17,21)(18,20)",
                perm"(1,3,4,9)(2,6,7,13)(5,10,11,17)(8,14,15,20)(12,18,19,23)(16,21,22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([
                perm"(1,2,4,7)(3,6,9,13)(5,16,11,22)(8,19,15,12)(10,21,17,24)(14,23,20,18)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,13)(8,14)(11,17)(12,18)(15,20)(16,21)(19,23)(22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([
                perm"(1,2)(3,13)(4,7)(5,16)(6,9)(8,12)(10,24)(11,22)(14,23)(15,19)(17,21)(18,20)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,13)(8,14)(11,17)(12,18)(15,20)(16,21)(19,23)(22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5)", perm"(6,7,8,9)"]),
            PermGroup([
                perm"(1,2)(3,14)(4,7)(5,8)(6,10)(9,21)(11,15)(12,16)(13,18)(17,24)(19,22)(20,23)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,13)(8,14)(11,17)(12,18)(15,20)(16,21)(19,23)(22,24)",
                perm"(1,4,11)(2,7,15)(3,9,17)(5,12,19)(6,13,20)(8,16,22)(10,18,23)(14,21,24)",
                perm"(1,5)(2,8)(3,10)(4,12)(6,14)(7,16)(9,18)(11,19)(13,21)(15,22)(17,23)(20,24)",
            ]),
            PermGroup([
                perm"(1,2,5,8)(3,14,10,6)(4,7,12,16)(9,21,18,13)(11,15,19,22)(17,24,23,20)",
                perm"(1,3,5,10)(2,6,8,14)(4,9,12,18)(7,13,16,21)(11,17,19,23)(15,20,22,24)",
                perm"(1,4,11)(2,7,15)(3,9,17)(5,12,19)(6,13,20)(8,16,22)(10,18,23)(14,21,24)",
                perm"(1,5)(2,8)(3,10)(4,12)(6,14)(7,16)(9,18)(11,19)(13,21)(15,22)(17,23)(20,24)",
            ]),
            PermGroup([
                perm"(1,2)(3,13)(4,8)(5,7)(6,9)(10,21)(11,20)(12,16)(14,18)(15,17)(19,24)(22,23)",
                perm"(1,3,9)(2,6,13)(4,11,23)(5,19,17)(7,15,24)(8,22,20)(10,18,12)(14,21,16)",
                perm"(1,4)(2,7)(3,10)(5,12)(6,14)(8,16)(9,17)(11,19)(13,20)(15,22)(18,23)(21,24)",
                perm"(1,5)(2,8)(3,11)(4,12)(6,15)(7,16)(9,18)(10,19)(13,21)(14,22)(17,23)(20,24)",
            ]),
            PermGroup([
                perm"(1,2)(3,6)(4,7)(5,8)(9,13)(10,14)(11,15)(12,16)(17,20)(18,21)(19,22)(23,24)",
                perm"(1,3,9)(2,6,13)(4,11,23)(5,19,17)(7,15,24)(8,22,20)(10,18,12)(14,21,16)",
                perm"(1,4)(2,7)(3,10)(5,12)(6,14)(8,16)(9,17)(11,19)(13,20)(15,22)(18,23)(21,24)",
                perm"(1,5)(2,8)(3,11)(4,12)(6,15)(7,16)(9,18)(10,19)(13,21)(14,22)(17,23)(20,24)",
            ]),
            PermGroup([
                perm"(1,2)(3,6)(4,7)(5,16)(8,12)(9,13)(10,21)(11,22)(14,18)(15,19)(17,24)(20,23)",
                perm"(1,3)(2,6)(4,9)(5,10)(7,13)(8,14)(11,17)(12,18)(15,20)(16,21)(19,23)(22,24)",
                perm"(1,4)(2,7)(3,9)(5,11)(6,13)(8,15)(10,17)(12,19)(14,20)(16,22)(18,23)(21,24)",
                perm"(1,5,12)(2,8,16)(3,10,18)(4,11,19)(6,14,21)(7,15,22)(9,17,23)(13,20,24)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6)", perm"(7,8,9)"]),
        ],
        [
            PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25)"]),
            PermGroup([perm"(1,2,3,4,5)", perm"(6,7,8,9,10)"]),
        ],
        [
            PermGroup([
                perm"(1,2)(3,26)(4,25)(5,24)(6,23)(7,22)(8,21)(9,20)(10,19)(11,18)(12,17)(13,16)(14,15)",
                perm"(1,3,5,7,9,11,13,15,17,19,21,23,25)(2,4,6,8,10,12,14,16,18,20,22,24,26)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5,6,7,8,9,10,11,12,13,14,15)"]),
        ],
        [
            PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)"]),
            PermGroup([perm"(1,2,3)", perm"(4,5,6,7,8,9,10,11,12)"]),
            PermGroup([
                perm"(1,2,5)(3,14,25)(4,7,12)(6,19,17)(8,26,24)(9,22,11)(10,15,20)(13,27,16)(18,23,21)",
                perm"(1,3,8)(2,6,13)(4,9,16)(5,11,18)(7,14,21)(10,17,23)(12,19,24)(15,22,26)(20,25,27)",
                perm"(1,4,10)(2,7,15)(3,9,17)(5,12,20)(6,14,22)(8,16,23)(11,19,25)(13,21,26)(18,24,27)",
            ]),
            PermGroup([
                perm"(1,2,5,4,7,12,10,15,20)(3,14,25,9,22,11,17,6,19)(8,26,24,16,13,27,23,21,18)",
                perm"(1,3,8)(2,6,13)(4,9,16)(5,11,18)(7,14,21)(10,17,23)(12,19,24)(15,22,26)(20,25,27)",
                perm"(1,4,10)(2,7,15)(3,9,17)(5,12,20)(6,14,22)(8,16,23)(11,19,25)(13,21,26)(18,24,27)",
            ]),
            PermGroup([perm"(1,2,3)", perm"(4,5,6)", perm"(7,8,9)"]),
        ],
        [
            PermGroup([
                perm"(1,2,3,5)(4,26,7,28)(6,27,9,24)(8,22,11,25)(10,23,13,20)(12,18,15,21)(14,19,17,16)",
                perm"(1,3)(2,5)(4,7)(6,9)(8,11)(10,13)(12,15)(14,17)(16,19)(18,21)(20,23)(22,25)(24,27)(26,28)",
                perm"(1,4,8,12,16,20,24)(2,6,10,14,18,22,26)(3,7,11,15,19,23,27)(5,9,13,17,21,25,28)",
            ]),
            PermGroup([perm"(1,2,3,4)", perm"(5,6,7,8,9,10,11)"]),
            PermGroup([
                perm"(1,2)(3,5)(4,26)(6,24)(7,28)(8,22)(9,27)(10,20)(11,25)(12,18)(13,23)(14,16)(15,21)(17,19)",
                perm"(1,3)(2,5)(4,7)(6,9)(8,11)(10,13)(12,15)(14,17)(16,19)(18,21)(20,23)(22,25)(24,27)(26,28)",
                perm"(1,4,8,12,16,20,24)(2,6,10,14,18,22,26)(3,7,11,15,19,23,27)(5,9,13,17,21,25,28)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4)", perm"(5,6,7,8,9,10,11)"]),
        ],
        [PermGroup([perm"(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29)"])],
        [
            PermGroup([
                perm"(1,2)(3,5)(4,12)(6,9)(7,10)(8,18)(11,15)(13,16)(14,24)(17,21)(19,22)(20,28)(23,26)(25,30)(27,29)",
                perm"(1,3,7,13,19)(2,5,10,16,22)(4,8,14,20,25)(6,11,17,23,27)(9,15,21,26,29)(12,18,24,28,30)",
                perm"(1,4,9)(2,6,12)(3,8,15)(5,11,18)(7,14,21)(10,17,24)(13,20,26)(16,23,28)(19,25,29)(22,27,30)",
            ]),
            PermGroup([
                perm"(1,2)(3,5)(4,24)(6,21)(7,10)(8,28)(9,18)(11,26)(12,15)(13,30)(14,23)(16,29)(17,20)(19,27)(22,25)",
                perm"(1,3,7)(2,5,10)(4,8,13)(6,11,16)(9,14,19)(12,17,22)(15,20,25)(18,23,27)(21,26,29)(24,28,30)",
                perm"(1,4,9,15,21)(2,6,12,18,24)(3,8,14,20,26)(5,11,17,23,28)(7,13,19,25,29)(10,16,22,27,30)",
            ]),
            PermGroup([
                perm"(1,2)(3,10)(4,24)(5,7)(6,21)(8,30)(9,18)(11,29)(12,15)(13,28)(14,27)(16,26)(17,25)(19,23)(20,22)",
                perm"(1,3,7)(2,5,10)(4,8,13)(6,11,16)(9,14,19)(12,17,22)(15,20,25)(18,23,27)(21,26,29)(24,28,30)",
                perm"(1,4,9,15,21)(2,6,12,18,24)(3,8,14,20,26)(5,11,17,23,28)(7,13,19,25,29)(10,16,22,27,30)",
            ]),
            PermGroup([perm"(1,2)", perm"(3,4,5)", perm"(6,7,8,9,10)"]),
        ],
    ])
)
